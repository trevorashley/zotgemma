"""Zotero full-text access: ``.zotero-ft-cache`` files and BM25 over ``fulltext.sqlite``."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

from . import config

FTS_OPERATORS = {"and", "or", "not", "near"}
STOPWORDS = frozenset(
    "a an the of in on at to for from by with without and or not is are was were be been being as it its this that "
    "these those which who whom what when where how why do does did can could should would may might into onto "
    "about over under between among than then so such via using use used based their there".split()
)
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def sanitize_fts_query(query: str, drop_stopwords: bool = True) -> str:
    """Turn natural language into a safe FTS5 query.

    Splits into alphanumeric terms, drops FTS operators (AND/OR/NOT/NEAR) and,
    optionally, English stopwords, quotes each term, and joins with ``OR`` so
    BM25 ranks partial matches. Returns ``""`` if no usable term remains.
    """
    seen: list[str] = []
    for tok in _TOKEN_RE.findall(query.lower()):
        if tok in FTS_OPERATORS or tok in seen:
            continue
        seen.append(tok)
    if drop_stopwords:
        kept = [t for t in seen if t not in STOPWORDS]
        seen = kept or seen  # a query made only of stopwords still searches
    return " OR ".join(f'"{t}"' for t in seen)


def cache_path(attachment_key: str):
    """Path of the extracted-text cache for an attachment key."""
    return config.STORAGE_DIR / attachment_key / ".zotero-ft-cache"


def load_text(attachment_key: str) -> str:
    """Return the extracted text of an attachment, or ``""`` if no cache exists."""
    p = cache_path(attachment_key)
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def cache_mtime(attachment_key: str) -> float:
    """mtime of the text cache, or 0.0 if absent."""
    try:
        return cache_path(attachment_key).stat().st_mtime
    except OSError:
        return 0.0


@dataclass(slots=True)
class FulltextHit:
    """A full-text BM25 hit on an attachment (``rowid`` == attachment itemID)."""

    attachment_id: int
    score: float  # positive; larger is better


def _open_fulltext() -> sqlite3.Connection:
    if not config.FULLTEXT_SQLITE.exists():
        raise FileNotFoundError(f"Zotero fulltext database not found at {config.FULLTEXT_SQLITE}")
    return sqlite3.connect(f"file:{config.FULLTEXT_SQLITE}?mode=ro&immutable=1", uri=True)


def bm25(query: str, limit: int = 100) -> list[FulltextHit]:
    """BM25-ranked attachments from Zotero's ``fulltextContent`` FTS5 table."""
    match = sanitize_fts_query(query)
    if not match:
        return []
    try:
        conn = _open_fulltext()
    except (sqlite3.Error, FileNotFoundError) as e:
        raise RuntimeError(f"Cannot open Zotero full-text index: {e}") from e
    try:
        rows = conn.execute(
            "SELECT rowid, rank FROM fulltextContent WHERE fulltextContent MATCH ? ORDER BY rank LIMIT ?",
            (match, limit),
        ).fetchall()
    except sqlite3.OperationalError as e:
        raise RuntimeError(f"Full-text query failed for {match!r}: {e}") from e
    finally:
        conn.close()
    return [FulltextHit(int(r[0]), -float(r[1])) for r in rows]


def indexed_attachment_ids() -> set[int]:
    """Attachment IDs that have rows in Zotero's full-text index."""
    conn = _open_fulltext()
    try:
        return {r[0] for r in conn.execute("SELECT id FROM fulltextContent_docsize")}
    finally:
        conn.close()
