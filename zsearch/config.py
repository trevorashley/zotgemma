"""Static configuration: filesystem locations and model constants."""

from __future__ import annotations

from pathlib import Path

ZOTERO_DIR = Path.home() / "Zotero"
ZOTERO_SQLITE = ZOTERO_DIR / "zotero.sqlite"
FULLTEXT_SQLITE = ZOTERO_DIR / "fulltext.sqlite"
STORAGE_DIR = ZOTERO_DIR / "storage"
BBT_MIGRATED = ZOTERO_DIR / "better-bibtex.migrated"
BBT_RPC_URL = "http://localhost:23119/better-bibtex/json-rpc"

PROJECT_DIR = Path(__file__).resolve().parent.parent
INDEX_DB = PROJECT_DIR / "data" / "index.sqlite"
GOLDEN_YAML = PROJECT_DIR / "tests" / "golden.yaml"

MODEL_ID = "google/embeddinggemma-2"
EMBED_DIM = 768
SCHEMA_VERSION = "1"

# Documents without an abstract fall back to the start of the extracted text.
FALLBACK_TEXT_CHARS = 6000  # roughly 1,500 tokens
MAX_SEQ_LENGTH = 1024
EMBED_BATCH_SIZE = 16

RRF_K = 60
CANDIDATE_DEPTH = 100
