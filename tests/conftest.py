"""Synthetic Zotero 10 data directory and a stub embedder, so tests need neither Zotero nor the model."""

from __future__ import annotations

import hashlib
import re
import sqlite3
from pathlib import Path

import numpy as np
import platformdirs
import pytest

from zotgemma import config, citekeys, embedder, index, search

# Tables and columns zotgemma reads (trimmed from a real Zotero 10.0.5 database).
ZOTERO_DDL = """
CREATE TABLE version (schema TEXT PRIMARY KEY, version INT NOT NULL);
CREATE TABLE libraries (libraryID INTEGER PRIMARY KEY, type TEXT NOT NULL, editable INT NOT NULL,
    filesEditable INT NOT NULL, version INT NOT NULL DEFAULT 0, storageVersion INT NOT NULL DEFAULT 0,
    lastSync INT NOT NULL DEFAULT 0, archived INT NOT NULL DEFAULT 0);
CREATE TABLE groups (groupID INTEGER PRIMARY KEY, libraryID INT NOT NULL UNIQUE, name TEXT NOT NULL,
    description TEXT NOT NULL, version INT NOT NULL);
CREATE TABLE itemTypes (itemTypeID INTEGER PRIMARY KEY, typeName TEXT, templateItemTypeID INT, display INT DEFAULT 1);
CREATE TABLE items (itemID INTEGER PRIMARY KEY, itemTypeID INT NOT NULL,
    dateAdded TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, dateModified TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    clientDateModified TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, libraryID INT NOT NULL, key TEXT NOT NULL,
    version INT NOT NULL DEFAULT 0, synced INT NOT NULL DEFAULT 0, UNIQUE (libraryID, key));
CREATE TABLE fields (fieldID INTEGER PRIMARY KEY, fieldName TEXT, fieldFormatID INT);
CREATE TABLE itemDataValues (valueID INTEGER PRIMARY KEY, value UNIQUE);
CREATE TABLE itemData (itemID INT, fieldID INT, valueID, PRIMARY KEY (itemID, fieldID));
CREATE TABLE creatorTypes (creatorTypeID INTEGER PRIMARY KEY, creatorType TEXT);
CREATE TABLE creators (creatorID INTEGER PRIMARY KEY, firstName TEXT, lastName TEXT, fieldMode INT);
CREATE TABLE itemCreators (itemID INT NOT NULL, creatorID INT NOT NULL, creatorTypeID INT NOT NULL DEFAULT 1,
    orderIndex INT NOT NULL DEFAULT 0, PRIMARY KEY (itemID, creatorID, creatorTypeID, orderIndex));
CREATE TABLE tags (tagID INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);
CREATE TABLE itemTags (itemID INT NOT NULL, tagID INT NOT NULL, type INT NOT NULL, PRIMARY KEY (itemID, tagID));
CREATE TABLE collections (collectionID INTEGER PRIMARY KEY, collectionName TEXT NOT NULL, parentCollectionID INT,
    clientDateModified TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, libraryID INT NOT NULL, key TEXT NOT NULL,
    version INT NOT NULL DEFAULT 0, synced INT NOT NULL DEFAULT 0);
CREATE TABLE collectionItems (collectionID INT NOT NULL, itemID INT NOT NULL, orderIndex INT NOT NULL DEFAULT 0,
    PRIMARY KEY (collectionID, itemID));
CREATE TABLE itemAttachments (itemID INTEGER PRIMARY KEY, parentItemID INT, linkMode INT, contentType TEXT,
    charsetID INT, path TEXT, syncState INT DEFAULT 0, storageModTime INT, storageHash TEXT);
CREATE TABLE itemAnnotations (itemID INTEGER PRIMARY KEY, parentItemID INT NOT NULL, type INTEGER NOT NULL,
    authorName TEXT, text TEXT, comment TEXT, color TEXT, pageLabel TEXT, sortIndex TEXT NOT NULL,
    position TEXT NOT NULL, isExternal INT NOT NULL);
CREATE TABLE deletedItems (itemID INTEGER PRIMARY KEY, dateDeleted DEFAULT CURRENT_TIMESTAMP NOT NULL);
"""

FIELDS = ["title", "abstractNote", "date", "publicationTitle", "DOI", "url", "extra"]
TYPES = ["journalArticle", "book", "attachment", "note", "annotation"]

USER_KEYS = {"paper1": "AAAAAAA1", "paper2": "AAAAAAA2", "pdf": "PDFAAAA1", "html": "HTMLAAA1", "ann": "ANNAAAA1",
             "trash": "TRASHAA1", "group": "GROUPAA1", "gpdf": "GPDFAAA1"}
GROUP_ID = 4242
GROUP_LIBRARY_ID = 2

TEXTS = {
    "PDFAAAA1": "Abstract\nPassivity based control of robotic manipulators using Lyapunov functions and energy shaping.",
    "HTMLAAA1": "Consensus protocols for multi-agent networks: a web snapshot about graph Laplacian convergence.",
    "GPDFAAA1": "Group library paper on sampling-based motion planning with rapidly exploring random trees.",
}


def build_zotero_dir(root: Path, userdata: int = 130) -> Path:
    """Create a synthetic Zotero data dir under ``root`` and return it."""
    root.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(root / "zotero.sqlite")
    conn.executescript(ZOTERO_DDL)
    conn.execute("INSERT INTO version VALUES ('userdata', ?)", (userdata,))
    conn.executemany("INSERT INTO libraries (libraryID, type, editable, filesEditable) VALUES (?,?,1,1)",
                     [(1, "user"), (GROUP_LIBRARY_ID, "group")])
    conn.execute("INSERT INTO groups VALUES (?,?,?,?,0)", (GROUP_ID, GROUP_LIBRARY_ID, "Lab Group", ""))
    for n, t in enumerate(TYPES, 1):
        conn.execute("INSERT INTO itemTypes (itemTypeID, typeName) VALUES (?,?)", (n, t))
    fid = {}
    for n, f in enumerate(FIELDS, 1):
        conn.execute("INSERT INTO fields (fieldID, fieldName) VALUES (?,?)", (n, f))
        fid[f] = n
    conn.execute("INSERT INTO creatorTypes VALUES (1, 'author')")
    tid = {t: n for n, t in enumerate(TYPES, 1)}
    k = USER_KEYS
    # (itemID, type, library, key)
    rows = [(1, "journalArticle", 1, k["paper1"]), (2, "book", 1, k["paper2"]), (3, "attachment", 1, k["pdf"]),
            (4, "attachment", 1, k["html"]), (5, "annotation", 1, k["ann"]), (6, "journalArticle", 1, k["trash"]),
            (7, "journalArticle", GROUP_LIBRARY_ID, k["group"]), (8, "attachment", GROUP_LIBRARY_ID, k["gpdf"])]
    for iid, typ, lib, key in rows:
        conn.execute("INSERT INTO items (itemID, itemTypeID, libraryID, key, dateModified) VALUES (?,?,?,?,?)",
                     (iid, tid[typ], lib, key, "2026-01-01 00:00:00"))
    data = {
        1: {"title": "Passivity-based control of manipulators", "abstractNote": "We study passivity and energy shaping for robot arms.",
            "date": "2004-05-01", "publicationTitle": "Journal of Control", "extra": "Citation Key: smith2004passivity\nOther: x"},
        2: {"title": "Consensus in multi-agent networks", "date": "2010", "publicationTitle": "Book Press"},
        6: {"title": "Trashed paper about lyapunov", "abstractNote": "Should never appear."},
        7: {"title": "Sampling-based motion planning", "abstractNote": "Rapidly exploring random trees for planning.",
            "date": "2019"},
    }
    vid = 0
    for iid, fs in data.items():
        for f, v in fs.items():
            vid += 1
            conn.execute("INSERT INTO itemDataValues VALUES (?,?)", (vid, v))
            conn.execute("INSERT INTO itemData VALUES (?,?,?)", (iid, fid[f], vid))
    conn.execute("INSERT INTO creators VALUES (1, 'Alice', 'Smith', 0)")
    conn.execute("INSERT INTO creators VALUES (2, 'Bob', 'Jones', 0)")
    conn.execute("INSERT INTO itemCreators VALUES (1, 1, 1, 0)")
    conn.execute("INSERT INTO itemCreators VALUES (7, 2, 1, 0)")
    conn.execute("INSERT INTO tags VALUES (1, 'robotics')")
    conn.execute("INSERT INTO itemTags VALUES (1, 1, 0)")
    conn.execute("INSERT INTO collections (collectionID, collectionName, libraryID, key) VALUES (1, 'Control', 1, 'COLLAAA1')")
    conn.execute("INSERT INTO collectionItems (collectionID, itemID) VALUES (1, 1)")
    conn.executemany(
        "INSERT INTO itemAttachments (itemID, parentItemID, linkMode, contentType, path) VALUES (?,?,?,?,?)",
        [(3, 1, 1, "application/pdf", "storage:paper.pdf"), (4, 2, 1, "text/html", "storage:snap.html"),
         (8, 7, 0, "application/pdf", "storage:g.pdf")])
    conn.execute("INSERT INTO itemAnnotations VALUES (5, 3, 1, NULL, 'energy shaping is key', 'check', '#ff0', '12', '00001', '{}', 0)")
    conn.execute("INSERT INTO deletedItems (itemID) VALUES (6)")
    conn.commit()
    conn.close()

    for key, text in TEXTS.items():
        d = root / "storage" / key
        d.mkdir(parents=True, exist_ok=True)
        (d / ".zotero-ft-cache").write_text(text)

    ft = sqlite3.connect(root / "fulltext.sqlite")
    try:  # contentless_delete needs SQLite >= 3.43; the real file uses it, plain contentless works for reads
        ft.execute("CREATE VIRTUAL TABLE fulltextContent USING fts5(text, tokenize='unicode61', content='', contentless_delete=1)")
    except sqlite3.OperationalError:
        ft.execute("CREATE VIRTUAL TABLE fulltextContent USING fts5(text, tokenize='unicode61', content='')")
    for iid, key in ((3, "PDFAAAA1"), (4, "HTMLAAA1"), (8, "GPDFAAA1")):
        ft.execute("INSERT INTO fulltextContent (rowid, text) VALUES (?,?)", (iid, TEXTS[key]))
    ft.commit()
    ft.close()
    return root


class StubEmbedder:
    """Deterministic bag-of-words hashing embedder: texts sharing words get similar unit vectors."""

    class info:  # noqa: N801 - mimics EmbedderInfo
        device, dtype, sanity = "cpu", "float32", "stub"

    @staticmethod
    def _vec(text: str) -> np.ndarray:
        v = np.zeros(config.EMBED_DIM, dtype=np.float32)
        for tok in re.findall(r"[a-z]+", text.lower()):
            seed = int.from_bytes(hashlib.sha256(tok.encode()).digest()[:8], "little")
            v += np.random.default_rng(seed).standard_normal(config.EMBED_DIM).astype(np.float32)
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)

    def embed_documents_raw(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        return np.vstack([self._vec(t) for t in texts])

    def count_tokens(self, texts: list[str]) -> int:
        return sum(len(t.split()) for t in texts)


@pytest.fixture()
def zotero_dir(tmp_path: Path) -> Path:
    return build_zotero_dir(tmp_path / "Zotero")


_CONFIG_NAMES = ("ZOTERO_DIR", "ZOTERO_SQLITE", "FULLTEXT_SQLITE", "STORAGE_DIR", "DATA_DIR_SOURCE",
                 "BASE_ATTACHMENT_PATH", "INDEX_DB", "INDEX_DB_IS_DEFAULT", "DEVICE", "BBT_RPC_URL", "LEGACY_INDEX_DB")


REAL_DATA_DIR = Path(platformdirs.user_data_dir("zotgemma"))  # captured before any test redirects HOME


@pytest.fixture(autouse=True)
def _hermetic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Nothing a test does may touch the real user data dir, HOME or the checkout's legacy index."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_DATA_HOME", str(home / ".local" / "share"))
    monkeypatch.setenv("ZOTGEMMA_INDEX_DB", str(tmp_path / "index.sqlite"))
    monkeypatch.delenv("ZOTGEMMA_DEVICE", raising=False)
    monkeypatch.delenv("ZOTGEMMA_BATCH_SIZE", raising=False)
    monkeypatch.setattr(config, "LEGACY_INDEX_DB", tmp_path / "no-such-legacy" / "index.sqlite")


@pytest.fixture(autouse=True)
def _restore_config():
    """configure() mutates module globals; put them back after every test."""
    from zotgemma import cli
    saved = {n: getattr(config, n) for n in _CONFIG_NAMES}
    err = cli._config_error
    yield
    for n, v in saved.items():
        setattr(config, n, v)
    cli._config_error = err


@pytest.fixture()
def env(zotero_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point zotgemma at the synthetic library, a temp index and the stub embedder; BBT is unreachable."""
    monkeypatch.setenv("ZOTGEMMA_ZOTERO_DIR", str(zotero_dir))
    monkeypatch.setattr(config, "BBT_RPC_URL", "http://127.0.0.1:9/better-bibtex/json-rpc")
    config.configure(zotero_dir=zotero_dir, index_db=tmp_path / "index.sqlite")
    stub = StubEmbedder()
    for mod in (index, search, embedder):
        monkeypatch.setattr(mod, "get_embedder", lambda: stub, raising=False)
    return zotero_dir
