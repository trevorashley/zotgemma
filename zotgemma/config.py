"""Configuration: filesystem locations (resolved at run time) and model constants.

Paths depend on the machine and are filled in by :func:`configure`, which the CLI calls once
at startup. Overridable via ``--zotero-dir`` / ``ZOTGEMMA_ZOTERO_DIR``, ``--index`` /
``ZOTGEMMA_INDEX_DB``, ``--device`` / ``ZOTGEMMA_DEVICE``, ``ZOTGEMMA_BATCH_SIZE``, ``ZOTGEMMA_BBT_URL``.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping
from pathlib import Path

import platformdirs

from . import discovery

BBT_RPC_URL = os.environ.get("ZOTGEMMA_BBT_URL", "http://localhost:23119/better-bibtex/json-rpc")

PROJECT_DIR = Path(__file__).resolve().parent.parent
LEGACY_INDEX_DB = PROJECT_DIR / "data" / "index.sqlite"  # pre-Phase-1.5 location (dev checkouts only)
GOLDEN_YAML = PROJECT_DIR / "tests" / "golden.yaml"

MODEL_ID = "google/embeddinggemma-2"
EMBED_DIM = 768
SCHEMA_VERSION = "1"
# Bump when the document-string recipe changes; abstract-less items are then re-hashed (and re-embedded if changed).
DOC_FORMAT = "4"

# Documents without an abstract fall back to the start of the extracted text.
FALLBACK_TEXT_CHARS = 3600  # roughly 900 tokens, so title + header + body fit in MAX_SEQ_LENGTH
ABSTRACT_SEARCH_CHARS = 10000  # an "Abstract" heading only counts this early (articles, monographs, reports)
FRONT_MATTER_SEARCH_CHARS = 60000  # how far into the text to look for Preface/Introduction/Contents (books)
MAX_SEQ_LENGTH = 1024

DEVICES = ("mps", "cuda", "cpu")
BATCH_SIZE_BY_DEVICE = {"mps": 16, "cuda": 64, "cpu": 8}

RRF_K = 60
CANDIDATE_DEPTH = 100

# Zotero 10.0.5 writes userdata=130 (read from a real zotero.sqlite on 2026-10-08). The userdata schema
# numbers of Zotero 6/7 are lower, but the exact number at which 10.0.0 started is not known, so refuse
# only databases clearly older than that: below 125. The hard requirement, Zotero's FTS5 fulltext.sqlite,
# is checked separately.
MIN_USERDATA_VERSION = 125

# --- resolved by configure() -------------------------------------------------------------------
ZOTERO_DIR: Path = Path.home() / "Zotero"
ZOTERO_SQLITE: Path = ZOTERO_DIR / "zotero.sqlite"
FULLTEXT_SQLITE: Path = ZOTERO_DIR / "fulltext.sqlite"
STORAGE_DIR: Path = ZOTERO_DIR / "storage"
DATA_DIR_SOURCE: str = "unresolved"
BASE_ATTACHMENT_PATH: Path | None = None
INDEX_DB: Path = Path.home() / ".local" / "share" / "zotgemma" / "index.sqlite"
INDEX_DB_IS_DEFAULT: bool = True
DEVICE: str | None = None  # None = auto (mps, then cuda, then cpu)


def index_path_for(zotero_dir: Path, data_home: Path | None = None) -> Path:
    """Default index location for a library: ``<user data dir>/zotgemma/<name>-<hash>/index.sqlite``.

    The hash of the resolved data-dir path keeps several libraries apart under one install.
    """
    base = data_home if data_home is not None else Path(platformdirs.user_data_dir("zotgemma"))
    resolved = str(Path(zotero_dir).expanduser().resolve())
    slug = re.sub(r"[^A-Za-z0-9]+", "-", Path(resolved).name).strip("-").lower() or "zotero"
    return base / f"{slug}-{hashlib.sha256(resolved.encode()).hexdigest()[:10]}" / "index.sqlite"


def default_batch_size(device: str, env: Mapping[str, str] | None = None) -> int:
    """Embedding batch size: ``ZOTGEMMA_BATCH_SIZE`` if set, else 16 (mps), 64 (cuda), 8 (cpu)."""
    env = os.environ if env is None else env
    raw = env.get("ZOTGEMMA_BATCH_SIZE")
    if raw:
        try:
            n = int(raw)
        except ValueError:
            raise ValueError(f"ZOTGEMMA_BATCH_SIZE must be a positive integer, got {raw!r}") from None
        if n < 1:
            raise ValueError(f"ZOTGEMMA_BATCH_SIZE must be a positive integer, got {raw!r}")
        return n
    return BATCH_SIZE_BY_DEVICE.get(device, 8)


def configure(zotero_dir: str | Path | None = None, index_db: str | Path | None = None,
              device: str | None = None) -> discovery.Discovered:
    """Resolve and install the run-time paths. Raises :class:`discovery.DiscoveryError`.

    ``None`` arguments fall through to the environment, then to defaults.
    """
    global ZOTERO_DIR, ZOTERO_SQLITE, FULLTEXT_SQLITE, STORAGE_DIR, DATA_DIR_SOURCE, BASE_ATTACHMENT_PATH
    global INDEX_DB, INDEX_DB_IS_DEFAULT
    found = discovery.discover_zotero_dir(zotero_dir)
    ZOTERO_DIR = found.path
    ZOTERO_SQLITE = ZOTERO_DIR / "zotero.sqlite"
    FULLTEXT_SQLITE = ZOTERO_DIR / "fulltext.sqlite"
    STORAGE_DIR = ZOTERO_DIR / "storage"
    DATA_DIR_SOURCE = found.source
    base = found.prefs.get("extensions.zotero.baseAttachmentPath")
    BASE_ATTACHMENT_PATH = Path(base).expanduser() if base else None

    idx = index_db or os.environ.get("ZOTGEMMA_INDEX_DB")
    INDEX_DB_IS_DEFAULT = not idx
    INDEX_DB = Path(idx).expanduser() if idx else index_path_for(ZOTERO_DIR)

    set_device(device)
    return found


def set_device(device: str | None = None) -> None:
    """Set the embedding device (``mps``/``cuda``/``cpu``); None reads ``ZOTGEMMA_DEVICE``, else auto."""
    global DEVICE
    dev = (device or os.environ.get("ZOTGEMMA_DEVICE") or "").strip().lower() or None
    if dev is not None and dev not in DEVICES:
        raise ValueError(f"device must be one of {', '.join(DEVICES)}, got {dev!r}")
    DEVICE = dev
