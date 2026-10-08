"""Read-only access to the Zotero SQLite database.

The live database is never opened for writing and never modified. Each read
works on a snapshot copy in a temp directory (database plus a non-empty WAL).
"""

from __future__ import annotations

import re
import shutil
import sqlite3
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from . import config, fulltext
from .models import Annotation, Attachment, Item

EXCLUDED_TYPES = ("attachment", "note", "annotation")


class ZoteroDBError(RuntimeError):
    """Raised when the Zotero database cannot be read."""


@contextmanager
def snapshot(src: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Yield a read-only connection to a temp copy of ``zotero.sqlite``.

    Copies the DB and, if non-empty, its ``-wal`` file so recent writes are
    visible. Never touches the original.
    """
    src = src or config.ZOTERO_SQLITE
    if not src.exists():
        raise ZoteroDBError(f"Zotero database not found at {src}. Is the Zotero data directory {config.ZOTERO_DIR} correct?")
    with tempfile.TemporaryDirectory(prefix="zotgemma-") as tmp:
        dst = Path(tmp) / "zotero.sqlite"
        try:
            shutil.copy2(src, dst)
            wal = src.with_name(src.name + "-wal")
            if wal.exists() and wal.stat().st_size > 0:
                shutil.copy2(wal, dst.with_name(dst.name + "-wal"))
        except OSError as e:
            raise ZoteroDBError(f"Could not copy Zotero database {src}: {e}") from e
        conn = sqlite3.connect(f"file:{dst}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()


def format_creator(last: str | None, first: str | None, field_mode: int | None) -> str:
    """Format a creator as ``Last, F.``; single-field names are returned as-is."""
    last = (last or "").strip()
    first = (first or "").strip()
    if field_mode == 1 or not first:
        return last or first
    parts = [p for p in re.split(r"[\s-]+", first) if p]
    initials = " ".join(p[0].upper() + "." for p in parts)
    return f"{last}, {initials}" if last else initials


def parse_year(date: str | None) -> int | None:
    """Extract a 4-digit year from a Zotero date string (``2004-00-00 2004``)."""
    if not date:
        return None
    m = re.match(r"\s*(\d{4})", date)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(1[5-9]\d\d|20\d\d)\b", date)
    return int(m.group(1)) if m else None


def resolve_attachment_path(key: str, path: str | None) -> tuple[str | None, str | None]:
    """Return (filename, absolute path) for a ``storage:<file>`` attachment path."""
    if not path:
        return None, None
    if path.startswith("storage:"):
        name = path[len("storage:"):]
        return name, str(config.STORAGE_DIR / key / name)
    return Path(path).name, path  # linked file (absolute path)


def _field_map(conn: sqlite3.Connection, names: tuple[str, ...]) -> dict[int, dict[str, str]]:
    """Map itemID -> {fieldName: value} for the requested fields."""
    q = ",".join("?" * len(names))
    rows = conn.execute(
        f"""SELECT d.itemID, f.fieldName, v.value FROM itemData d
            JOIN fields f ON f.fieldID = d.fieldID
            JOIN itemDataValues v ON v.valueID = d.valueID
            WHERE f.fieldName IN ({q})""",
        names,
    )
    out: dict[int, dict[str, str]] = {}
    for r in rows:
        out.setdefault(r["itemID"], {})[r["fieldName"]] = r["value"]
    return out


def _creators(conn: sqlite3.Connection) -> dict[int, str]:
    """Map itemID -> formatted creator string (authors; editors if no authors)."""
    rows = conn.execute(
        """SELECT ic.itemID, ct.creatorType, c.lastName, c.firstName, c.fieldMode
           FROM itemCreators ic JOIN creators c ON c.creatorID = ic.creatorID
           JOIN creatorTypes ct ON ct.creatorTypeID = ic.creatorTypeID
           ORDER BY ic.itemID, ic.orderIndex"""
    )
    by_item: dict[int, dict[str, list[str]]] = {}
    for r in rows:
        by_item.setdefault(r["itemID"], {}).setdefault(r["creatorType"], []).append(
            format_creator(r["lastName"], r["firstName"], r["fieldMode"])
        )
    out: dict[int, str] = {}
    for iid, groups in by_item.items():
        names = groups.get("author") or groups.get("editor") or next(iter(groups.values()))
        out[iid] = "; ".join(n for n in names if n)
    return out


def _collection_paths(conn: sqlite3.Connection) -> dict[int, list[str]]:
    """Map itemID -> list of full collection paths ("A / B / C")."""
    cols = {r["collectionID"]: (r["collectionName"], r["parentCollectionID"])
            for r in conn.execute("SELECT collectionID, collectionName, parentCollectionID FROM collections")}

    def path(cid: int) -> str:
        parts: list[str] = []
        seen: set[int] = set()
        while cid is not None and cid in cols and cid not in seen:
            seen.add(cid)
            name, cid = cols[cid]
            parts.append(name)
        return " / ".join(reversed(parts))

    out: dict[int, list[str]] = {}
    for r in conn.execute("SELECT collectionID, itemID FROM collectionItems"):
        out.setdefault(r["itemID"], []).append(path(r["collectionID"]))
    return out


def _tags(conn: sqlite3.Connection) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for r in conn.execute("SELECT it.itemID, t.name FROM itemTags it JOIN tags t ON t.tagID = it.tagID"):
        out.setdefault(r["itemID"], []).append(r["name"])
    return out


def _pdf_attachments(conn: sqlite3.Connection) -> list[Attachment]:
    """All non-deleted PDF attachments with linkMode in {0 imported, 1 imported URL}."""
    rows = conn.execute(
        """SELECT a.itemID, i.key, a.parentItemID, a.path FROM itemAttachments a
           JOIN items i ON i.itemID = a.itemID
           WHERE a.contentType = 'application/pdf' AND a.linkMode IN (0, 1)
             AND a.itemID NOT IN (SELECT itemID FROM deletedItems)
           ORDER BY a.itemID"""
    )
    out = []
    for r in rows:
        fn, p = resolve_attachment_path(r["key"], r["path"])
        out.append(Attachment(r["itemID"], r["key"], r["parentItemID"], fn, p))
    return out


def best_attachment(atts: list[Attachment]) -> Attachment | None:
    """Pick the PDF with the largest ``.zotero-ft-cache``; ties and no-cache fall back to lowest itemID."""
    if not atts:
        return None

    def size(a: Attachment) -> int:
        try:
            return fulltext.cache_path(a.key).stat().st_size
        except OSError:
            return 0

    return max(atts, key=lambda a: (size(a), -a.item_id))


def load_items(conn: sqlite3.Connection) -> list[Item]:
    """Load all bibliographic items plus standalone PDFs as :class:`Item` records."""
    q = ",".join("?" * len(EXCLUDED_TYPES))
    rows = conn.execute(
        f"""SELECT i.itemID, i.key, t.typeName, i.dateModified FROM items i
            JOIN itemTypes t ON t.itemTypeID = i.itemTypeID
            WHERE t.typeName NOT IN ({q})
              AND i.itemID NOT IN (SELECT itemID FROM deletedItems)
            ORDER BY i.itemID""",
        EXCLUDED_TYPES,
    ).fetchall()
    fields = _field_map(conn, ("title", "abstractNote", "date", "publicationTitle", "proceedingsTitle",
                               "bookTitle", "publisher", "repository", "websiteTitle", "DOI", "url"))
    creators, tags, colls = _creators(conn), _tags(conn), _collection_paths(conn)
    pdfs = _pdf_attachments(conn)

    by_parent: dict[int, list[Attachment]] = {}
    for a in pdfs:
        if a.parent_item_id is not None:
            by_parent.setdefault(a.parent_item_id, []).append(a)

    items: list[Item] = []
    for r in rows:
        f = fields.get(r["itemID"], {})
        venue = (f.get("publicationTitle") or f.get("proceedingsTitle") or f.get("bookTitle")
                 or f.get("repository") or f.get("websiteTitle") or f.get("publisher") or "")
        items.append(Item(
            item_id=r["itemID"], key=r["key"], item_type=r["typeName"],
            title=f.get("title", "").strip(), year=parse_year(f.get("date")),
            authors=creators.get(r["itemID"], ""), venue=venue, abstract=f.get("abstractNote", "").strip(),
            doi=f.get("DOI", ""), url=f.get("url", ""), date_modified=r["dateModified"],
            tags=sorted(tags.get(r["itemID"], [])), collections=sorted(colls.get(r["itemID"], [])),
            attachment=best_attachment(by_parent.get(r["itemID"], [])),
            attachments=by_parent.get(r["itemID"], []),
        ))

    # Standalone PDFs (no parent) become their own items, titled from the filename.
    dates = {r["itemID"]: r["dateModified"] for r in conn.execute("SELECT itemID, dateModified FROM items")}
    for a in pdfs:
        if a.parent_item_id is not None:
            continue
        f = fields.get(a.item_id, {})
        title = f.get("title") or (Path(a.filename).stem.replace("_", " ") if a.filename else a.key)
        items.append(Item(
            item_id=a.item_id, key=a.key, item_type="attachment", title=title.strip(), year=parse_year(f.get("date")),
            authors=creators.get(a.item_id, ""), venue="", abstract="", doi="", url=f.get("url", ""),
            date_modified=dates.get(a.item_id, ""), tags=sorted(tags.get(a.item_id, [])),
            collections=sorted(colls.get(a.item_id, [])), attachment=a, attachments=[a], standalone=True,
        ))
    return items


def load_annotations(conn: sqlite3.Connection, attachment_ids: list[int]) -> list[Annotation]:
    """Annotations belonging to the given attachment item IDs, in reading order."""
    if not attachment_ids:
        return []
    q = ",".join("?" * len(attachment_ids))
    rows = conn.execute(
        f"""SELECT itemID, pageLabel, text, comment FROM itemAnnotations
            WHERE parentItemID IN ({q}) AND itemID NOT IN (SELECT itemID FROM deletedItems)
            ORDER BY parentItemID, sortIndex""",
        attachment_ids,
    )
    return [Annotation(r["itemID"], r["pageLabel"], r["text"] or "", r["comment"] or "") for r in rows]


def library_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Headline counts for ``zotgemma status``."""
    one = lambda sql: conn.execute(sql).fetchone()[0]  # noqa: E731
    live = "itemID NOT IN (SELECT itemID FROM deletedItems)"
    return {
        "bibliographic_items": one(
            f"""SELECT count(*) FROM items i JOIN itemTypes t USING(itemTypeID)
                WHERE t.typeName NOT IN ('attachment','note','annotation') AND i.{live}"""),
        "pdf_attachments": one(
            f"SELECT count(*) FROM itemAttachments WHERE contentType='application/pdf' AND {live}"),
        "annotations": one(f"SELECT count(*) FROM itemAnnotations WHERE {live}"),
        "collections": one("SELECT count(*) FROM collections"),
        "tags": one("SELECT count(*) FROM tags"),
    }
