"""Golden-set evaluation: recall@k for dense/keyword/hybrid and truncation dims."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import yaml

from . import config
from . import embedder
from .search import load_matrix, search

KS = (5, 10)
# (label, mode, truncate_dim)
CONFIGS: list[tuple[str, str, int]] = [
    ("keyword", "keyword", 768),
    ("dense@768", "dense", 768),
    ("dense@512", "dense", 512),
    ("dense@256", "dense", 256),
    ("hybrid@768", "hybrid", 768),
    ("hybrid@512", "hybrid", 512),
    ("hybrid@256", "hybrid", 256),
]


@dataclass(slots=True)
class GoldenQuery:
    """One golden-set entry."""

    query: str
    expected: list[str]  # Zotero item keys; any one is an acceptable answer
    kind: str
    note: str = ""


def load_golden(path: Path | None = None) -> list[GoldenQuery]:
    """Parse a golden-set YAML file (default: ``tests/golden.yaml`` if it exists)."""
    path = path or config.GOLDEN_YAML
    data = yaml.safe_load(path.read_text())
    return [GoldenQuery(e["query"], list(e["expected"]), e.get("kind", ""), e.get("note", "")) for e in data["queries"]]


def recall_at_k(ranked_keys: list[str], expected: list[str], k: int) -> tuple[float, bool]:
    """Return (fraction of expected keys in the top k, whether any was)."""
    top = set(ranked_keys[:k])
    found = sum(1 for e in expected if e in top)
    return found / len(expected), found > 0


def run_eval(conn: sqlite3.Connection, golden: list[GoldenQuery]) -> dict:
    """Run every config over the golden set.

    Returns ``{label: {"recall@5", "recall@10", "hit@5", "hit@10", "by_kind": {kind: hit@10}, "misses": [...]}}``.
    ``recall@k`` is the mean fraction of expected items found; ``hit@k`` is the share of
    queries with at least one expected item found.
    """
    emb = embedder.get_embedder()
    qvecs = [emb.embed_query(g.query) for g in golden]
    matrix = load_matrix(conn)
    results: dict[str, dict] = {}
    for label, mode, dim in CONFIGS:
        rec = {k: [] for k in KS}
        hit = {k: [] for k in KS}
        by_kind: dict[str, list[bool]] = {}
        misses = []
        for g, qv in zip(golden, qvecs):
            hits = search(conn, g.query, limit=max(KS), mode=mode, dim=dim, matrix=matrix, qvec=qv)
            keys = [h.key for h in hits]
            for k in KS:
                r, h = recall_at_k(keys, g.expected, k)
                rec[k].append(r); hit[k].append(h)
            ok10 = hit[10][-1]
            by_kind.setdefault(g.kind, []).append(ok10)
            if not ok10:
                misses.append(g.query)
        n = len(golden)
        results[label] = {
            **{f"recall@{k}": sum(rec[k]) / n for k in KS},
            **{f"hit@{k}": sum(hit[k]) / n for k in KS},
            "by_kind": {kd: sum(v) / len(v) for kd, v in by_kind.items()},
            "misses": misses,
        }
    return results
