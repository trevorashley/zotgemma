"""End-to-end tests against the synthetic Zotero 10 library (no Zotero, model or network)."""

from __future__ import annotations

import json
import sqlite3

import pytest
from typer.testing import CliRunner

from conftest import GROUP_ID, REAL_DATA_DIR, USER_KEYS, build_zotero_dir
from zotgemma import citekeys, config, discovery, embedder, index, search, zotero_db
from zotgemma.cli import app
from zotgemma.models import zotero_link


def _items(zotero_dir):
    with zotero_db.snapshot() as z:
        return zotero_db.load_items(z)


def test_load_items_all_libraries_and_exclusions(env):
    items = {i.key: i for i in _items(env)}
    k = USER_KEYS
    assert set(items) == {k["paper1"], k["paper2"], k["group"]}  # trashed item, annotation and attachments excluded
    assert items[k["group"]].library_id == 2 and items[k["group"]].group_id == GROUP_ID
    assert items[k["paper1"]].group_id is None and items[k["paper1"]].library_name == "My Library"
    assert items[k["paper1"]].authors == "Smith, A."
    assert items[k["paper1"]].extra_citekey == "smith2004passivity"
    assert items[k["paper1"]].collections == ["Control"]


def test_html_attachment_counts_as_text_attachment(env):
    items = {i.key: i for i in _items(env)}
    att = items[USER_KEYS["paper2"]].attachment
    assert att is not None and att.content_type == "text/html"
    with zotero_db.snapshot() as z:
        atts = [a for i in zotero_db.load_items(z) for a in i.attachments]
    assert zotero_db.attachment_type_counts(atts) == {"application/pdf": 2, "text/html": 1}


def test_annotations_and_schema_guard(env, tmp_path):
    with zotero_db.snapshot() as z:
        anns = zotero_db.load_annotations(z, [3])
    assert [(a.page_label, a.text) for a in anns] == [("12", "energy shaping is key")]
    old = build_zotero_dir(tmp_path / "old", userdata=100)
    with pytest.raises(zotero_db.ZoteroDBError, match="userdata schema version 100"):
        with zotero_db.snapshot(old / "zotero.sqlite"):
            pass


def test_sync_is_incremental_and_keyword_search_works(env):
    s = index.sync()
    assert (s.total, s.embedded) == (3, 3)
    assert s.citekey_source == citekeys.CACHED_SOURCE and s.citekeys_from_extra == 1
    assert index.sync().embedded == 0
    conn = index.open_index()
    try:
        hits = search.search(conn, "passivity manipulators", mode="keyword")
        assert hits[0].key == USER_KEYS["paper1"] and hits[0].citekey == "smith2004passivity"
        assert hits[0].link == f"zotero://select/library/items/{USER_KEYS['paper1']}"
        # full-text-only match (word only in the HTML snapshot text cache)
        assert search.search(conn, "laplacian", mode="keyword")[0].key == USER_KEYS["paper2"]
        assert all(h.key != USER_KEYS["trash"] for h in search.search(conn, "lyapunov", mode="keyword"))
    finally:
        conn.close()


def test_group_links_and_library_filter(env):
    index.sync()
    conn = index.open_index()
    try:
        hits = search.search(conn, "sampling planning", mode="keyword")
        g = next(h for h in hits if h.key == USER_KEYS["group"])
        assert g.link == f"zotero://select/groups/{GROUP_ID}/items/{USER_KEYS['group']}"
        assert [h.key for h in search.search(conn, "planning control", mode="keyword", library="user")
                if h.group_id is not None] == []
        only = search.search(conn, "planning control passivity", mode="keyword", library=str(GROUP_ID))
        assert {h.key for h in only} == {USER_KEYS["group"]}
        assert {h.key for h in search.search(conn, "sampling", mode="keyword", library="lab")} == {USER_KEYS["group"]}
    finally:
        conn.close()


def test_hybrid_and_dense_with_stub_embedder(env):
    index.sync()
    conn = index.open_index()
    try:
        hybrid = search.search(conn, "passivity energy shaping robot arms", mode="hybrid")
        assert hybrid[0].key == USER_KEYS["paper1"] and "dense" in hybrid[0].ranks
        dense = search.search(conn, "rapidly exploring random trees", mode="dense")
        assert dense[0].key == USER_KEYS["group"]
    finally:
        conn.close()


def test_cached_citekeys_survive_when_zotero_is_down(env, monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(citekeys, "fetch_bbt", lambda refs, **kw: {(1, USER_KEYS["paper2"]): "jones2010consensus"})
        assert index.sync().citekey_source == "bbt"
    s = index.sync()  # BBT unreachable again
    assert s.citekey_source == citekeys.CACHED_SOURCE
    conn = index.open_index()
    try:
        assert conn.execute("SELECT citekey FROM items WHERE key=?", (USER_KEYS["paper2"],)).fetchone()[0] == "jones2010consensus"
    finally:
        conn.close()


def test_legacy_index_schema_migrates_without_reembedding(env):
    index.sync()
    conn = index.open_index()
    # simulate a pre-Phase-1.5 items table (UNIQUE(key), no library columns)
    conn.executescript("""
        ALTER TABLE items RENAME TO items_new;
        DROP INDEX items_citekey; DROP INDEX items_attachment;
        CREATE TABLE items (item_id INTEGER PRIMARY KEY, key TEXT NOT NULL UNIQUE, citekey TEXT, item_type TEXT, title TEXT,
            year INTEGER, authors TEXT, venue TEXT, abstract TEXT, doi TEXT, url TEXT, tags TEXT, collections TEXT,
            date_modified TEXT, fulltext_mtime REAL, doc_text_hash TEXT, attachment_id INTEGER,
            attachment_key TEXT, attachment_path TEXT, standalone INTEGER DEFAULT 0);
        INSERT INTO items SELECT item_id, key, citekey, item_type, title, year, authors, venue, abstract, doi, url, tags,
            collections, date_modified, fulltext_mtime, doc_text_hash, attachment_id, attachment_key, attachment_path, standalone
            FROM items_new;
        DROP TABLE items_new;""")
    conn.close()
    assert index.sync().embedded == 0
    conn = index.open_index()
    assert "library_id" in {r[1] for r in conn.execute("PRAGMA table_info(items)")}
    conn.close()


def test_cli_status_search_json_and_eval_golden(env, tmp_path):
    runner = CliRunner()
    assert runner.invoke(app, ["index"]).exit_code == 0
    r = runner.invoke(app, ["status"])
    assert r.exit_code == 0 and "text/html" in r.output and "application/pdf" in r.output
    r = runner.invoke(app, ["search", "passivity", "--mode", "keyword", "--json"])
    data = json.loads(r.output)
    assert data[0]["key"] == USER_KEYS["paper1"] and data[0]["link"].startswith("zotero://select/library/items/")
    r = runner.invoke(app, ["search", "planning", "--mode", "keyword", "--library", "nosuchlib"])
    assert r.exit_code == 1
    golden = tmp_path / "g.yaml"
    golden.write_text(f"queries:\n  - query: passivity manipulators\n    expected: [{USER_KEYS['paper1']}]\n    kind: concept\n")
    r = runner.invoke(app, ["eval", "--golden", str(golden)])
    assert r.exit_code == 0 and "hybrid@768" in r.output


def test_cli_version_and_missing_dir(monkeypatch, tmp_path):
    runner = CliRunner()
    assert runner.invoke(app, ["--version"]).output.startswith("zotgemma ")
    monkeypatch.setenv("ZOTGEMMA_ZOTERO_DIR", str(tmp_path / "nonexistent"))
    r = runner.invoke(app, ["status"])
    assert r.exit_code == 1 and "ZOTGEMMA_ZOTERO_DIR" in r.output and "zotero.sqlite" in r.output


def test_zotero_link_forms():
    assert zotero_link("K1") == "zotero://select/library/items/K1"
    assert zotero_link("K1", 99) == "zotero://select/groups/99/items/K1"


def _real_listing():
    return sorted(str(p) for p in REAL_DATA_DIR.rglob("*")) if REAL_DATA_DIR.exists() else []


def _default_location(env, monkeypatch):
    """Reconfigure so the index is at the (HOME-redirected) default location."""
    monkeypatch.delenv("ZOTGEMMA_INDEX_DB")
    config.configure(zotero_dir=env)
    assert config.INDEX_DB_IS_DEFAULT
    return config.INDEX_DB


def test_legacy_index_copied_only_when_it_belongs_to_this_library(env, monkeypatch, tmp_path):
    legacy = tmp_path / "legacy" / "index.sqlite"
    legacy.parent.mkdir()
    index.sync(index_path=legacy)  # an index built from this library
    monkeypatch.setattr(config, "LEGACY_INDEX_DB", legacy)
    dest = _default_location(env, monkeypatch)
    assert str(dest).startswith(str(tmp_path / "home"))
    assert index.migrate_legacy_index() == legacy and dest.is_file()
    # a legacy index of some other library (keys absent from this zotero.sqlite) is not adopted
    dest.unlink()
    conn = sqlite3.connect(legacy)
    conn.execute("UPDATE items SET key = 'ZZ' || key")
    conn.commit(); conn.close()
    assert index.migrate_legacy_index() is None and not dest.exists()


def test_tests_create_nothing_outside_tmp(env, monkeypatch, tmp_path):
    before = _real_listing()
    runner = CliRunner()
    monkeypatch.delenv("ZOTGEMMA_INDEX_DB")
    assert runner.invoke(app, ["index"]).exit_code == 0
    assert config.INDEX_DB.is_relative_to(tmp_path)
    assert runner.invoke(app, ["status"]).exit_code == 0
    assert _real_listing() == before
    assert not list(tmp_path.rglob("models--*")) and not (tmp_path / "hf").exists()  # no HF cache created


def test_model_loading_fails_fast_in_tests():
    with pytest.raises(RuntimeError, match="tests must not load the model"):
        embedder.Embedder(device="cpu")


def test_sync_survives_item_id_change_with_same_key(env):
    index.sync()
    conn = index.open_index()
    conn.execute("UPDATE items SET item_id = item_id + 1000 WHERE key = ?", (USER_KEYS["paper1"],))
    conn.execute("UPDATE vec_items SET item_id = item_id + 1000 WHERE item_id = 1")
    conn.commit(); conn.close()
    s = index.sync(force=True)
    assert s.removed == 1 and s.embedded == 3


def test_numeric_library_spec_falls_back_to_name(env, tmp_path):
    index.sync()
    conn = index.open_index()
    try:
        assert search.library_clause(str(GROUP_ID), conn)[0] == "group_id = ?"
        conn.execute("UPDATE items SET library_name = 'Group 2024' WHERE group_id IS NOT NULL")
        assert search.library_clause("2024", conn)[0].startswith("lower(library_name)")
    finally:
        conn.close()
