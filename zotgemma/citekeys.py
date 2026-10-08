"""Better BibTeX citation keys: live JSON-RPC with an offline SQLite fallback."""

from __future__ import annotations

import logging
import sqlite3

import httpx

from . import config

log = logging.getLogger(__name__)
BATCH = 500  # the endpoint accepted 1,243 keys in one call when tested; stay conservative


class CitekeyError(RuntimeError):
    """Raised when the BBT endpoint cannot be used."""


def fetch_bbt(keys: list[str], batch: int = BATCH, timeout: float = 30.0) -> dict[str, str]:
    """Fetch citation keys for Zotero item keys via Better BibTeX JSON-RPC.

    Raises :class:`CitekeyError` if Zotero/BBT is unreachable or returns an error.
    Items without a key are simply absent from the result.
    """
    out: dict[str, str] = {}
    for i in range(0, len(keys), batch):
        chunk = keys[i:i + batch]
        try:
            r = httpx.post(config.BBT_RPC_URL, timeout=timeout, json={
                "jsonrpc": "2.0", "method": "item.citationkey", "params": [chunk], "id": 1})
            r.raise_for_status()
            data = r.json()
        except (httpx.HTTPError, ValueError) as e:
            raise CitekeyError(
                f"Better BibTeX endpoint {config.BBT_RPC_URL} unavailable ({e}). Is Zotero running with BBT installed?"
            ) from e
        if "error" in data:
            raise CitekeyError(f"Better BibTeX returned an error: {data['error']}")
        out.update({k: v for k, v in (data.get("result") or {}).items() if v})
    return out


def fetch_migrated() -> dict[str, str]:
    """Read keys from the (possibly stale) ``better-bibtex.migrated`` snapshot."""
    if not config.BBT_MIGRATED.exists():
        raise CitekeyError(f"Fallback file {config.BBT_MIGRATED} does not exist")
    try:
        conn = sqlite3.connect(f"file:{config.BBT_MIGRATED}?mode=ro&immutable=1", uri=True)
        try:
            return {r[0]: r[1] for r in conn.execute("SELECT itemKey, citationKey FROM citationkey")}
        finally:
            conn.close()
    except sqlite3.Error as e:
        raise CitekeyError(f"Cannot read {config.BBT_MIGRATED}: {e}") from e


def get_citekeys(keys: list[str]) -> tuple[dict[str, str], str]:
    """Return ``(mapping, source)`` where source is ``"bbt"`` or ``"migrated"``.

    Uses the live endpoint when possible; otherwise the migrated snapshot.
    """
    try:
        return fetch_bbt(keys), "bbt"
    except CitekeyError as e:
        log.warning("%s; falling back to %s", e, config.BBT_MIGRATED.name)
        m = fetch_migrated()
        return {k: m[k] for k in keys if k in m}, "migrated"


def bbt_available(timeout: float = 2.0) -> bool:
    """True if the BBT JSON-RPC endpoint answers."""
    try:
        r = httpx.post(config.BBT_RPC_URL, timeout=timeout, json={
            "jsonrpc": "2.0", "method": "item.citationkey", "params": [[]], "id": 1})
        return r.status_code == 200 and "error" not in r.json()
    except (httpx.HTTPError, ValueError):
        return False
