"""Better BibTeX citation keys via the live JSON-RPC endpoint (cached in the index when Zotero is down)."""

from __future__ import annotations

import logging

import httpx

from . import config

log = logging.getLogger(__name__)
BATCH = 500  # the endpoint accepted 1,243 keys in one call when tested; stay conservative
CACHED_SOURCE = "cached (Zotero not running)"

LibKey = tuple[int, str]  # (Zotero libraryID, item key)


class CitekeyError(RuntimeError):
    """Raised when the BBT endpoint cannot be used."""


def rpc_ref(library_id: int, key: str) -> str:
    """BBT's ``libraryID:itemKey`` form. A bare key means the user library only, so always qualify."""
    return f"{library_id}:{key}"


def fetch_bbt(refs: list[LibKey], batch: int = BATCH, timeout: float = 30.0) -> dict[LibKey, str]:
    """Fetch citation keys for ``(libraryID, itemKey)`` pairs via Better BibTeX JSON-RPC.

    Raises :class:`CitekeyError` if Zotero/BBT is unreachable or returns an error.
    Items without a key are simply absent from the result.
    """
    out: dict[LibKey, str] = {}
    for i in range(0, len(refs), batch):
        chunk = refs[i:i + batch]
        by_ref = {rpc_ref(lib, key): (lib, key) for lib, key in chunk}
        try:
            r = httpx.post(config.BBT_RPC_URL, timeout=timeout, json={
                "jsonrpc": "2.0", "method": "item.citationkey", "params": [list(by_ref)], "id": 1})
            r.raise_for_status()
            data = r.json()
        except (httpx.HTTPError, ValueError) as e:
            raise CitekeyError(
                f"Better BibTeX endpoint {config.BBT_RPC_URL} unavailable ({e}). Is Zotero running with BBT installed?"
            ) from e
        if "error" in data:
            raise CitekeyError(f"Better BibTeX returned an error: {data['error']}")
        for ref, ck in (data.get("result") or {}).items():
            if ck and ref in by_ref:
                out[by_ref[ref]] = ck
    return out


def get_citekeys(refs: list[LibKey]) -> tuple[dict[LibKey, str], str]:
    """Return ``(mapping, source)``: ``"bbt"`` from the live endpoint, or ``({}, CACHED_SOURCE)`` if it is down."""
    try:
        return fetch_bbt(refs), "bbt"
    except CitekeyError as e:
        log.info("%s; keeping cached cite keys", e)
        return {}, CACHED_SOURCE


def bbt_available(timeout: float = 2.0) -> bool:
    """True if the BBT JSON-RPC endpoint answers."""
    try:
        r = httpx.post(config.BBT_RPC_URL, timeout=timeout, json={
            "jsonrpc": "2.0", "method": "item.citationkey", "params": [[]], "id": 1})
        return r.status_code == 200 and "error" not in r.json()
    except (httpx.HTTPError, ValueError):
        return False
