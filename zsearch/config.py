"""Static configuration: filesystem locations and model constants."""

from __future__ import annotations

import os
from pathlib import Path

# Overridable via ZSEARCH_ZOTERO_DIR, ZSEARCH_BBT_URL, ZSEARCH_INDEX_DB.
ZOTERO_DIR = Path(os.environ.get("ZSEARCH_ZOTERO_DIR", Path.home() / "Zotero")).expanduser()
ZOTERO_SQLITE = ZOTERO_DIR / "zotero.sqlite"
FULLTEXT_SQLITE = ZOTERO_DIR / "fulltext.sqlite"
STORAGE_DIR = ZOTERO_DIR / "storage"
BBT_MIGRATED = ZOTERO_DIR / "better-bibtex.migrated"
BBT_RPC_URL = os.environ.get("ZSEARCH_BBT_URL", "http://localhost:23119/better-bibtex/json-rpc")

PROJECT_DIR = Path(__file__).resolve().parent.parent
INDEX_DB = Path(os.environ.get("ZSEARCH_INDEX_DB", PROJECT_DIR / "data" / "index.sqlite")).expanduser()
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
EMBED_BATCH_SIZE = 16

RRF_K = 60
CANDIDATE_DEPTH = 100
