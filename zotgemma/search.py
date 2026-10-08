"""Dense + BM25 + reciprocal-rank-fusion search over the index."""

from __future__ import annotations

import logging
import re
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import sqlite_vec

from . import config, fulltext
from . import embedder
from .embedder import truncate
from .models import Hit

MODES = ("dense", "keyword", "hybrid")
log = logging.getLogger(__name__)


def parse_year_range(spec: str | None) -> tuple[int | None, int | None]:
    """Parse ``2018``, ``2018:``, ``:2005`` or ``2010:2020`` into (from, to)."""
    if not spec:
        return None, None
    spec = spec.strip()
    if ":" not in spec:
        y = int(spec)
        return y, y
    lo, hi = spec.split(":", 1)
    return (int(lo) if lo.strip() else None, int(hi) if hi.strip() else None)


def rrf_fuse(rankings: Mapping[str, Sequence[int]], k: int = config.RRF_K) -> list[tuple[int, float, dict[str, int]]]:
    """Reciprocal rank fusion.

    ``rankings`` maps a source name to an ordered list of ids (best first). Each
    id scores ``sum(1 / (k + rank))`` over sources (rank is 1-based). Returns
    ``(id, score, {source: rank})`` sorted by descending score; ties break by id
    for determinism.
    """
    scores: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}
    for src, ids in rankings.items():
        for r, i in enumerate(ids, start=1):
            if src in ranks.get(i, {}):
                continue  # ignore duplicates within one list
            scores[i] = scores.get(i, 0.0) + 1.0 / (k + r)
            ranks.setdefault(i, {})[src] = r
    return [(i, scores[i], ranks[i]) for i in sorted(scores, key=lambda x: (-scores[x], x))]


@dataclass(slots=True)
class DenseMatrix:
    """All item vectors in memory, for truncation experiments and brute-force search."""

    ids: np.ndarray  # (n,)
    vecs: np.ndarray  # (n, 768), unit-norm


def load_matrix(conn: sqlite3.Connection) -> DenseMatrix:
    """Load every stored vector from ``vec_items``."""
    rows = conn.execute("SELECT item_id, embedding FROM vec_items").fetchall()
    ids = np.array([r[0] for r in rows], dtype=np.int64)
    vecs = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows]) if rows else np.zeros((0, config.EMBED_DIM), np.float32)
    return DenseMatrix(ids, vecs)


def library_clause(spec: str, conn: sqlite3.Connection | None = None) -> tuple[str, list]:
    """SQL condition + args for ``--library``: ``user``, a group ID, or a group-name substring.

    A numeric spec is a group ID if some indexed library has that ID (needs ``conn``; without it,
    numeric means ID); otherwise it is matched as a name substring.
    """
    s = spec.strip().lower()
    if s in ("user", "my", "me", "personal"):
        return "group_id IS NULL", []
    if s.isdigit() and (conn is None or conn.execute("SELECT 1 FROM items WHERE group_id = ? LIMIT 1", (int(s),)).fetchone()):
        return "group_id = ?", [int(s)]
    return "lower(library_name) LIKE ?", [f"%{s}%"]


def list_libraries(conn: sqlite3.Connection) -> list[tuple[int | None, str, int]]:
    """``(group_id or None, name, item_count)`` for each library present in the index."""
    return [(r[0], r[1], r[2]) for r in conn.execute(
        "SELECT group_id, library_name, count(*) FROM items GROUP BY library_id ORDER BY library_id")]


def _allowed_ids(conn: sqlite3.Connection, year_from: int | None, year_to: int | None,
                 item_type: str | None, collection: str | None, library: str | None = None) -> set[int] | None:
    """Item IDs passing the filters, or None if no filter is active."""
    where, args = [], []
    if library:
        clause, largs = library_clause(library, conn)
        where.append(clause); args.extend(largs)
    if year_from is not None:
        where.append("year >= ?"); args.append(year_from)
    if year_to is not None:
        where.append("year <= ?"); args.append(year_to)
    if item_type:
        where.append("item_type = ?"); args.append(item_type)
    if collection:
        where.append("lower(collections) LIKE ?"); args.append(f"%{collection.lower()}%")
    if not where:
        return None
    return {r[0] for r in conn.execute(f"SELECT item_id FROM items WHERE {' AND '.join(where)}", args)}


def dense_ranking(conn: sqlite3.Connection, qvec: np.ndarray, depth: int, allowed: set[int] | None = None,
                  dim: int = config.EMBED_DIM, matrix: DenseMatrix | None = None) -> list[tuple[int, float]]:
    """Ranked ``(item_id, cosine)`` pairs. Uses vec0 at full dimension, numpy otherwise."""
    if dim == config.EMBED_DIM and matrix is None:
        total = conn.execute("SELECT count(*) FROM vec_items").fetchone()[0]
        k = total if allowed is not None else min(depth, total)
        rows = conn.execute("SELECT item_id, distance FROM vec_items WHERE embedding MATCH ? AND k = ?",
                            (sqlite_vec.serialize_float32(qvec.tolist()), max(k, 1))).fetchall()
        out = [(r[0], 1.0 - r[1]) for r in rows if allowed is None or r[0] in allowed]
        return out[:depth]
    matrix = matrix or load_matrix(conn)
    q = truncate(qvec[None, :], dim)[0]
    sims = truncate(matrix.vecs, dim) @ q
    order = np.argsort(-sims)
    out = []
    for j in order:
        i = int(matrix.ids[j])
        if allowed is None or i in allowed:
            out.append((i, float(sims[j])))
            if len(out) >= depth:
                break
    return out


def meta_ranking(conn: sqlite3.Connection, query: str, depth: int) -> list[int]:
    """BM25 over title/authors/abstract/tags (our ``item_fts``)."""
    match = fulltext.sanitize_fts_query(query)
    if not match:
        return []
    rows = conn.execute(
        "SELECT rowid FROM item_fts WHERE item_fts MATCH ? ORDER BY bm25(item_fts, 6.0, 3.0, 1.0, 2.0) LIMIT ?",
        (match, depth)).fetchall()
    return [r[0] for r in rows]


def fulltext_ranking(conn: sqlite3.Connection, query: str, depth: int) -> list[int]:
    """Zotero full-text BM25, rolled up from attachments to items (best attachment wins)."""
    att_to_item = {r[0]: r[1] for r in conn.execute("SELECT attachment_id, item_id FROM attachments")}
    out: list[int] = []
    seen: set[int] = set()
    for h in fulltext.bm25(query, depth * 3):
        i = att_to_item.get(h.attachment_id)
        if i is not None and i not in seen:
            seen.add(i)
            out.append(i)
    return out


def search(conn: sqlite3.Connection, query: str, limit: int = 10, mode: str = "hybrid",
           year_from: int | None = None, year_to: int | None = None, item_type: str | None = None,
           collection: str | None = None, library: str | None = None, dim: int = config.EMBED_DIM,
           matrix: DenseMatrix | None = None, qvec: np.ndarray | None = None) -> list[Hit]:
    """Search the library.

    ``mode`` is ``dense``, ``keyword`` (item-metadata BM25 + Zotero full-text BM25, fused)
    or ``hybrid`` (dense list + the fused keyword list, fused by RRF with k=60). Filters apply to every
    candidate list before fusion. ``dim`` truncates dense vectors (Matryoshka) for experiments.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if not query.strip():
        return []
    allowed = _allowed_ids(conn, year_from, year_to, item_type, collection, library)
    depth = config.CANDIDATE_DEPTH if allowed is None else 10_000

    def keep(ids: list[int]) -> list[int]:
        out = ids if allowed is None else [i for i in ids if i in allowed]
        return out[: config.CANDIDATE_DEPTH]

    rankings: dict[str, list[int]] = {}
    dense_scores: dict[int, float] = {}
    if mode in ("dense", "hybrid"):
        if qvec is None:
            qvec = embedder.get_embedder().embed_query(query)
        pairs = dense_ranking(conn, qvec, config.CANDIDATE_DEPTH, allowed, dim, matrix)
        rankings["dense"] = [i for i, _ in pairs]
        dense_scores = dict(pairs)
    if mode in ("keyword", "hybrid"):
        rankings["meta"] = keep(meta_ranking(conn, query, depth))
        try:
            rankings["fulltext"] = keep(fulltext_ranking(conn, query, depth))
        except RuntimeError as e:  # Zotero may be rebuilding its index; keep going on metadata + dense
            log.warning("Zotero full-text index unavailable, searching without it: %s", e)
            rankings["fulltext"] = []

    detail: dict[str, dict[str, int]] = {}  # per-item sub-source ranks, for transparency
    if mode == "hybrid":
        # Fuse the two keyword lists into one first so dense and keyword carry equal weight.
        meta, ft = rankings.pop("meta"), rankings.pop("fulltext")
        for src, ids in (("meta", meta), ("fulltext", ft)):
            for r, i in enumerate(ids, 1):
                detail.setdefault(i, {})[src] = r
        rankings["keyword"] = [i for i, _, _ in rrf_fuse({"meta": meta, "fulltext": ft})][: config.CANDIDATE_DEPTH]
    fused = rrf_fuse(rankings)[:limit]
    if detail:
        fused = [(i, sc, {**r, **detail.get(i, {})}) for i, sc, r in fused]
    if mode == "dense":
        fused = [(i, dense_scores[i], r) for i, _, r in fused]
    return _hydrate(conn, fused)


def _hydrate(conn: sqlite3.Connection, fused: list[tuple[int, float, dict[str, int]]]) -> list[Hit]:
    hits: list[Hit] = []
    for i, score, ranks in fused:
        r = conn.execute("SELECT key, citekey, title, authors, year, item_type, venue, library_id, group_id, library_name "
            "FROM items WHERE item_id = ?", (i,)).fetchone()
        if r is None:
            continue
        hits.append(Hit(i, r["key"], r["citekey"], r["title"], r["authors"], r["year"], r["item_type"], r["venue"], score, ranks,
                        r["library_id"], r["group_id"], r["library_name"]))
    return hits


_AUTHOR_SPLIT = re.compile(r"\s*;\s*")


def short_authors(authors: str, n: int = 2) -> str:
    """Abbreviate an author string to the first ``n`` names plus ``et al.``."""
    names = [a for a in _AUTHOR_SPLIT.split(authors) if a]
    if len(names) <= n:
        return "; ".join(names)
    return "; ".join(names[:n]) + " et al."
