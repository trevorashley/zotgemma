"""Index database: schema, incremental sync from Zotero, deletion."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import sqlite_vec

from . import config, citekeys, fulltext, zotero_db
from .embedder import build_document, get_embedder
from .models import Item

log = logging.getLogger(__name__)

_ITEMS_BODY = """(
    item_id INTEGER PRIMARY KEY, key TEXT NOT NULL, library_id INTEGER NOT NULL DEFAULT 1, group_id INTEGER,
    library_name TEXT NOT NULL DEFAULT '', citekey TEXT, item_type TEXT, title TEXT,
    year INTEGER, authors TEXT, venue TEXT, abstract TEXT, doi TEXT, url TEXT, tags TEXT, collections TEXT,
    date_modified TEXT, fulltext_mtime REAL, doc_text_hash TEXT, attachment_id INTEGER,
    attachment_key TEXT, attachment_path TEXT, standalone INTEGER DEFAULT 0,
    UNIQUE (library_id, key)
)"""

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS items {_ITEMS_BODY};

CREATE INDEX IF NOT EXISTS items_citekey ON items(citekey);
CREATE INDEX IF NOT EXISTS items_attachment ON items(attachment_id);
CREATE VIRTUAL TABLE IF NOT EXISTS item_fts USING fts5(title, authors, abstract, tags);
CREATE VIRTUAL TABLE IF NOT EXISTS vec_items USING vec0(
    item_id INTEGER PRIMARY KEY, embedding float[{config.EMBED_DIM}] distance_metric=cosine);
CREATE TABLE IF NOT EXISTS attachments (
    attachment_id INTEGER PRIMARY KEY, item_id INTEGER NOT NULL, key TEXT NOT NULL, path TEXT,
    has_text INTEGER NOT NULL DEFAULT 0, text_bytes INTEGER NOT NULL DEFAULT 0, is_best INTEGER NOT NULL DEFAULT 0,
    library_id INTEGER NOT NULL DEFAULT 1, content_type TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS attachments_item ON attachments(item_id);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
"""

TAG_SEP = "\n"


def migrate_legacy_index(dest: Path | None = None) -> Path | None:
    """Copy a pre-Phase-1.5 ``data/index.sqlite`` to the platformdirs location, once.

    Only when ``dest`` is the default location, does not exist yet, and the library is the one the
    old code would have used (``ZOTGEMMA_ZOTERO_DIR`` or ``~/Zotero``), so a 10-minute embedding run is
    not repeated. The legacy file is left in place. Returns the legacy path if it was copied.
    """
    dest = dest or config.INDEX_DB
    legacy = config.LEGACY_INDEX_DB
    if not config.INDEX_DB_IS_DEFAULT or dest.exists() or not legacy.is_file():
        return None
    old_dir = Path(os.environ.get("ZOTGEMMA_ZOTERO_DIR") or Path.home() / "Zotero").expanduser()
    if old_dir.resolve() != config.ZOTERO_DIR.resolve() or not _legacy_matches_library(legacy):
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(legacy, dest)
    log.warning("copied existing index %s -> %s (the old file can be deleted)", legacy, dest)
    return legacy


def _legacy_matches_library(legacy: Path, sample: int = 25) -> bool:
    """True if a sample of the legacy index's item keys all exist in the current Zotero library."""
    try:
        lc = sqlite3.connect(f"file:{legacy}?mode=ro", uri=True)
        try:
            keys = [r[0] for r in lc.execute("SELECT key FROM items ORDER BY random() LIMIT ?", (sample,))]
        finally:
            lc.close()
        if not keys:
            return False
        with zotero_db.snapshot() as z:
            q = ",".join("?" * len(keys))
            found = z.execute(f"SELECT count(DISTINCT key) FROM items WHERE key IN ({q})", keys).fetchone()[0]
        return found == len(set(keys))
    except (sqlite3.Error, zotero_db.ZoteroDBError):
        return False


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Upgrade an index built before Phase 1.5 in place (no re-embedding; vectors are keyed by item_id).

    The ``items`` rebuild is a single transaction, so a failure midway leaves the old table intact.
    """
    cols = {r[1] for r in conn.execute("PRAGMA table_info(items)")}
    if cols and "library_id" not in cols:
        new_cols = [c for c in _NEW_ITEM_COLS if c in cols]
        shared = ", ".join(new_cols)
        conn.isolation_level = None  # manual transaction control
        try:
            conn.execute("BEGIN")
            conn.execute(f"CREATE TABLE items_new {_ITEMS_BODY}")
            conn.execute(f"INSERT INTO items_new ({shared}) SELECT {shared} FROM items")
            conn.execute("DROP INDEX IF EXISTS items_citekey")
            conn.execute("DROP INDEX IF EXISTS items_attachment")
            conn.execute("DROP TABLE items")
            conn.execute("ALTER TABLE items_new RENAME TO items")
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.isolation_level = ""
    acols = {r[1] for r in conn.execute("PRAGMA table_info(attachments)")}
    if acols and "library_id" not in acols:
        conn.execute("ALTER TABLE attachments ADD COLUMN library_id INTEGER NOT NULL DEFAULT 1")
    if acols and "content_type" not in acols:
        conn.execute("ALTER TABLE attachments ADD COLUMN content_type TEXT NOT NULL DEFAULT ''")
    conn.commit()


_NEW_ITEM_COLS = ("item_id", "key", "citekey", "item_type", "title", "year", "authors", "venue", "abstract", "doi",
                  "url", "tags", "collections", "date_modified", "fulltext_mtime", "doc_text_hash", "attachment_id",
                  "attachment_key", "attachment_path", "standalone")


def open_index(path: Path | None = None, create: bool = False) -> sqlite3.Connection:
    """Open the index DB with sqlite-vec loaded. Raises FileNotFoundError if absent and not ``create``."""
    if path is None:
        path = config.INDEX_DB
        migrate_legacy_index(path)
    if not path.exists():
        if not create:
            raise FileNotFoundError(f"No index at {path}. Run `zotgemma index` first.")
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        conn.enable_load_extension(True)
    except AttributeError:
        conn.close()
        raise RuntimeError(
            "This Python's sqlite3 cannot load extensions (sqlite-vec needs that). The macOS system Python "
            "lacks it; use a uv-managed or Homebrew Python, e.g. `uv tool install --python 3.12 ...`.") from None
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    _migrate_schema(conn)
    conn.executescript(SCHEMA)  # idempotent; adds tables introduced after the index was built
    return conn


def get_meta(conn: sqlite3.Connection, k: str) -> str | None:
    """Read a value from the ``meta`` table."""
    r = conn.execute("SELECT v FROM meta WHERE k = ?", (k,)).fetchone()
    return r[0] if r else None


def set_meta(conn: sqlite3.Connection, k: str, v: str) -> None:
    """Write a value to the ``meta`` table."""
    conn.execute("INSERT INTO meta (k, v) VALUES (?, ?) ON CONFLICT(k) DO UPDATE SET v = excluded.v", (k, v))


# An abstract heading: "Abstract", "Abstract This monograph...", "Abstract—...", "ABSTRACT:".
# The word must end the line, be followed by punctuation, or be followed by a capitalised sentence;
# lines with dot leaders or a trailing page number (contents entries) are rejected. This keeps
# "abstract features" (a figure caption) and "Abstract Factory ... 525" from counting.
_ABSTRACT_RE = re.compile(
    r"^[ \t\f]*abstract(?:[ \t]*$|[ \t]*[:.\-\u2014\u2013]|[ \t]+(?=(?-i:[A-Z])))"
    r"(?![^\n]*(?:\.{4,}|(?:\.[ \t]+){3,}))(?![^\n]*\d[ \t]*$)",
    re.I | re.M,
)
_HEADING_RES = [
    re.compile(rf"^[ \t\f]*(?:\d+\.?[ \t]+)?(?:{w})[ \t]*$", re.I | re.M)
    for w in ("preface", "introduction", "contents|table of contents")
]


def skip_front_matter(text: str) -> str:
    """Drop series/editor boilerplate before the real content.

    Priority: an ``Abstract`` heading within the first ``ABSTRACT_SEARCH_CHARS`` characters
    (articles, monographs, reports), else the first bare ``Preface``, ``Introduction`` or
    ``Contents`` line within ``FRONT_MATTER_SEARCH_CHARS`` (books). The short window for
    ``Abstract`` stops chapter abstracts and body text deep inside a book from winning over
    its preface. Returns ``text`` unchanged if nothing matches.
    """
    m = _ABSTRACT_RE.search(text[: config.ABSTRACT_SEARCH_CHARS])
    if m:
        return text[m.start():].lstrip()
    head = text[: config.FRONT_MATTER_SEARCH_CHARS]
    for rx in _HEADING_RES:
        m = rx.search(head)
        if m:
            return text[m.start():].lstrip()
    return text


def build_item_body(authors: str, year: int | None, venue: str, abstract: str, fallback_text: str = "") -> str:
    """Body of the document string: ``{authors} ({year}). {venue}. {abstract-or-text}``.

    Falls back to the first ``FALLBACK_TEXT_CHARS`` characters of ``fallback_text``
    (whitespace-collapsed) when there is no abstract. Empty parts are omitted.
    """
    content = abstract.strip() or re.sub(r"\s+", " ", skip_front_matter(fallback_text)).strip()[: config.FALLBACK_TEXT_CHARS]
    head = authors.strip()
    if year:
        head = f"{head} ({year})" if head else f"({year})"
    parts = [p.rstrip(".") + "." for p in (head, venue.strip()) if p]
    if content:
        parts.append(content)
    return " ".join(parts)


def item_document(item: Item, fallback_text: str = "") -> str:
    """Full ``title: ... | text: ...`` document string for an item."""
    return build_document(item.title, build_item_body(item.authors, item.year, item.venue, item.abstract, fallback_text))


def is_unchanged(old, item: Item, mtime: float, has_vec: bool, force: bool) -> bool:
    """True if the stored row proves the vector is current (no re-read or re-embed needed).

    A NULL ``doc_text_hash`` means an earlier sync was interrupted between writing metadata and
    storing the vector, so the vector cannot be trusted.
    """
    return bool(old is not None and not force and has_vec and old["doc_text_hash"] is not None
                and old["date_modified"] == item.date_modified
                and (item.abstract or old["fulltext_mtime"] == mtime))


def needs_recheck(old, item: Item, mtime: float, has_vec: bool, force: bool, recheck_fallback: bool) -> bool:
    """True if the item's document must be rebuilt and re-hashed this sync.

    That is whenever :func:`is_unchanged` fails, or when the document recipe for abstract-less
    items changed (``recheck_fallback``) and this item has no abstract.
    """
    return not is_unchanged(old, item, mtime, has_vec, force) or (recheck_fallback and not item.abstract)


def doc_hash(doc: str) -> str:
    """Stable hash of a document string, used to skip re-embedding."""
    return hashlib.sha256(doc.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class SyncStats:
    """Outcome of :func:`sync`."""

    total: int = 0
    embedded: int = 0
    removed: int = 0
    tokens: int = 0
    embed_seconds: float = 0.0
    seconds: float = 0.0
    citekey_source: str = ""
    citekeys_found: int = 0
    citekeys_from_extra: int = 0
    dtype: str = ""
    device: str = ""


def sync(progress: Callable[[int, int], None] | None = None, force: bool = False,
         index_path: Path | None = None, commit_every: int = 64) -> SyncStats:
    """Bring the index in line with Zotero. Embeds only new/changed items.

    ``progress(done, total)`` is called while embedding. ``force`` re-embeds everything.
    """
    t0 = time.time()
    stats = SyncStats()
    with zotero_db.snapshot() as zconn:
        items = zotero_db.load_items(zconn)
    stats.total = len(items)

    conn = open_index(index_path, create=True)
    try:
        if get_meta(conn, "model_id") not in (None, config.MODEL_ID) or get_meta(conn, "schema_version") not in (None, config.SCHEMA_VERSION):
            force = True
        recheck_fallback = get_meta(conn, "doc_format") != config.DOC_FORMAT
        existing = {r["item_id"]: r for r in conn.execute(
            "SELECT item_id, date_modified, fulltext_mtime, doc_text_hash, citekey FROM items")}
        have_vec = {r[0] for r in conn.execute("SELECT item_id FROM vec_items")}

        # Remove vanished items first: an item that kept its key under a new itemID (library reset,
        # re-added group) would otherwise collide with its stale row on UNIQUE(library_id, key).
        present = {it.item_id for it in items}
        gone = [i for i in existing if i not in present]
        for i in gone:
            conn.execute("DELETE FROM items WHERE item_id = ?", (i,))
            conn.execute("DELETE FROM vec_items WHERE item_id = ?", (i,))
            have_vec.discard(i)
        conn.commit()
        stats.removed = len(gone)

        todo: list[tuple[Item, str, str]] = []  # (item, doc string, hash)
        hashes: dict[int, str] = {}
        mtimes: dict[int, float] = {}
        for it in items:
            att_key = it.attachment.key if it.attachment else None
            mtime = fulltext.cache_mtime(att_key) if att_key else 0.0
            mtimes[it.item_id] = mtime
            old = existing.get(it.item_id)
            if not needs_recheck(old, it, mtime, it.item_id in have_vec, force, recheck_fallback):
                hashes[it.item_id] = old["doc_text_hash"]
                continue
            text = "" if it.abstract or not att_key else fulltext.load_text(att_key)
            doc = item_document(it, text)
            h = doc_hash(doc)
            hashes[it.item_id] = h
            if old is None or force or it.item_id not in have_vec or old["doc_text_hash"] != h:
                todo.append((it, doc, h))

        # Embed in commit-sized slices so an interrupted run keeps its progress.
        if todo:
            emb = get_embedder()
            stats.device, stats.dtype = emb.info.device, emb.info.dtype
            # persist metadata first so vectors always have a row
            _upsert_items(conn, items, existing, hashes, mtimes, hash_override_ids={it.item_id for it, _, _ in todo})
            conn.commit()
            done = 0
            for i in range(0, len(todo), commit_every):
                sl = todo[i:i + commit_every]
                ts = time.time()
                vecs = emb.embed_documents_raw([d for _, d, _ in sl])
                stats.embed_seconds += time.time() - ts
                stats.tokens += emb.count_tokens([d for _, d, _ in sl])
                for (it, _, h), v in zip(sl, vecs):
                    conn.execute("DELETE FROM vec_items WHERE item_id = ?", (it.item_id,))
                    conn.execute("INSERT INTO vec_items (item_id, embedding) VALUES (?, ?)",
                                 (it.item_id, sqlite_vec.serialize_float32(v.tolist())))
                    conn.execute("UPDATE items SET doc_text_hash = ? WHERE item_id = ?", (h, it.item_id))
                conn.commit()
                done += len(sl)
                if progress:
                    progress(done, len(todo))
            stats.embedded = len(todo)
        _upsert_items(conn, items, existing, hashes, mtimes, hash_override_ids=set())

        _sync_attachments(conn, items)
        _refresh_citekeys(conn, items, stats)
        _rebuild_fts(conn)
        set_meta(conn, "model_id", config.MODEL_ID)
        set_meta(conn, "embed_dim", str(config.EMBED_DIM))
        set_meta(conn, "schema_version", config.SCHEMA_VERSION)
        set_meta(conn, "doc_format", config.DOC_FORMAT)
        set_meta(conn, "last_sync", time.strftime("%Y-%m-%d %H:%M:%S"))
        if stats.dtype:
            set_meta(conn, "dtype", stats.dtype)
        conn.commit()
    finally:
        conn.close()
    stats.seconds = time.time() - t0
    return stats


def _upsert_items(conn: sqlite3.Connection, items: list[Item], existing: dict, hashes: dict[int, str],
                  mtimes: dict[int, float], hash_override_ids: set[int]) -> None:
    """Insert/update metadata rows. Items about to be re-embedded keep an empty hash until their vector lands."""
    for it in items:
        a = it.attachment
        h = None if it.item_id in hash_override_ids else hashes.get(it.item_id)
        conn.execute(
            """INSERT INTO items (item_id, key, library_id, group_id, library_name, item_type, title, year, authors, venue, abstract, doi, url, tags,
                                  collections, date_modified, fulltext_mtime, doc_text_hash, attachment_id,
                                  attachment_key, attachment_path, standalone)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(item_id) DO UPDATE SET key=excluded.key, library_id=excluded.library_id,
                 group_id=excluded.group_id, library_name=excluded.library_name, item_type=excluded.item_type,
                 title=excluded.title, year=excluded.year, authors=excluded.authors, venue=excluded.venue,
                 abstract=excluded.abstract, doi=excluded.doi, url=excluded.url, tags=excluded.tags,
                 collections=excluded.collections, date_modified=excluded.date_modified,
                 fulltext_mtime=excluded.fulltext_mtime, doc_text_hash=excluded.doc_text_hash,
                 attachment_id=excluded.attachment_id, attachment_key=excluded.attachment_key,
                 attachment_path=excluded.attachment_path, standalone=excluded.standalone""",
            (it.item_id, it.key, it.library_id, it.group_id, it.library_name, it.item_type, it.title, it.year, it.authors, it.venue, it.abstract, it.doi, it.url,
             TAG_SEP.join(it.tags), TAG_SEP.join(it.collections), it.date_modified, mtimes.get(it.item_id, 0.0), h,
             a.item_id if a else None, a.key if a else None, a.path if a else None, int(it.standalone)),
        )
    conn.commit()


def _sync_attachments(conn: sqlite3.Connection, items: list[Item]) -> None:
    """Rebuild the ``attachments`` table: every non-deleted PDF or text-cached attachment of every item."""
    conn.execute("DELETE FROM attachments")
    for it in items:
        for a in it.attachments:
            try:
                size = fulltext.cache_path(a.key).stat().st_size
            except OSError:
                size = 0
            conn.execute("""INSERT OR REPLACE INTO attachments
                            (attachment_id, item_id, key, path, has_text, text_bytes, is_best, library_id, content_type)
                            VALUES (?,?,?,?,?,?,?,?,?)""",
                         (a.item_id, it.item_id, a.key, a.path, int(size > 0), size,
                          int(it.attachment is not None and it.attachment.item_id == a.item_id),
                          a.library_id, a.content_type))


def _refresh_citekeys(conn: sqlite3.Connection, items: list[Item], stats: SyncStats) -> None:
    """Update ``items.citekey``.

    Live Better BibTeX keys win. Items BBT has no key for fall back to a ``Citation Key:`` line in ``extra``.
    When Zotero is not running, cached keys are kept and ``extra`` only fills rows that have none.
    """
    mapping, source = citekeys.get_citekeys([(i.library_id, i.key) for i in items])
    stats.citekey_source = source
    live = source == "bbt"
    for (lib, key), ck in mapping.items():
        conn.execute("UPDATE items SET citekey = ? WHERE library_id = ? AND key = ?", (ck, lib, key))
    where = "" if live else " AND citekey IS NULL"
    for it in items:
        if it.extra_citekey and (it.library_id, it.key) not in mapping:
            cur = conn.execute(f"UPDATE items SET citekey = ? WHERE library_id = ? AND key = ?{where}",
                               (it.extra_citekey, it.library_id, it.key))
            stats.citekeys_from_extra += cur.rowcount
    stats.citekeys_found = conn.execute("SELECT count(*) FROM items WHERE citekey IS NOT NULL").fetchone()[0]


def _rebuild_fts(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM item_fts")
    conn.execute("""INSERT INTO item_fts (rowid, title, authors, abstract, tags)
                    SELECT item_id, title, authors, abstract, replace(tags, char(10), ' ') FROM items""")


def index_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Row counts for status output."""
    return {
        "items": conn.execute("SELECT count(*) FROM items").fetchone()[0],
        "vectors": conn.execute("SELECT count(*) FROM vec_items").fetchone()[0],
        "with_citekey": conn.execute("SELECT count(*) FROM items WHERE citekey IS NOT NULL").fetchone()[0],
        "with_abstract": conn.execute("SELECT count(*) FROM items WHERE abstract != ''").fetchone()[0],
    }
