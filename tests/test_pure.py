"""Tests for pure helper functions (no Zotero, model or network needed)."""

from zotgemma.embedder import build_document, truncate
from zotgemma.evaluation import recall_at_k
from zotgemma.fulltext import sanitize_fts_query
from zotgemma.index import build_item_body, item_document
from zotgemma.models import Item
from zotgemma.search import parse_year_range, rrf_fuse
from zotgemma.zotero_db import format_creator, parse_year

import numpy as np
import pytest

from pathlib import Path

from zotgemma import config, discovery
from zotgemma.embedder import pick_device
from zotgemma.models import Item
from zotgemma.zotero_db import parse_extra_citekey, resolve_attachment_path


# --- FTS sanitizer ---------------------------------------------------------
def test_sanitize_quotes_terms_and_ors():
    assert sanitize_fts_query("Lyapunov stability") == '"lyapunov" OR "stability"'


def test_sanitize_drops_operators_and_punctuation():
    q = sanitize_fts_query('foo AND bar OR NOT "baz" NEAR (qux)* -x:y')
    assert q == '"foo" OR "bar" OR "baz" OR "qux" OR "x" OR "y"'


def test_sanitize_drops_stopwords_but_keeps_all_stopword_query():
    assert sanitize_fts_query("the theory of the control") == '"theory" OR "control"'
    assert sanitize_fts_query("the of and") == '"the" OR "of"'


def test_sanitize_empty_and_dedup():
    assert sanitize_fts_query("  ?!  ") == ""
    assert sanitize_fts_query("lie lie") == '"lie"'


# --- RRF -------------------------------------------------------------------
def test_rrf_scores_and_order():
    out = rrf_fuse({"a": [1, 2, 3], "b": [3, 1]}, k=60)
    ids = [i for i, _, _ in out]
    assert ids[0] == 1  # 1/61 + 1/62 beats 3's 1/63 + 1/61
    s1 = dict((i, s) for i, s, _ in out)
    assert abs(s1[1] - (1 / 61 + 1 / 62)) < 1e-12
    ranks = {i: r for i, _, r in out}
    assert ranks[3] == {"a": 3, "b": 1}
    assert ranks[2] == {"a": 2}


def test_rrf_empty_and_single_list_and_duplicates():
    assert rrf_fuse({}) == []
    assert [i for i, _, _ in rrf_fuse({"a": [5, 4]})] == [5, 4]
    out = rrf_fuse({"a": [7, 7]})
    assert len(out) == 1 and abs(out[0][1] - 1 / 61) < 1e-12


def test_rrf_tie_breaks_by_id():
    out = rrf_fuse({"a": [2], "b": [1]})
    assert [i for i, _, _ in out] == [1, 2]


# --- document strings ------------------------------------------------------
def test_build_document_format():
    assert build_document("My Title", "body text") == "title: My Title | text: body text"
    assert build_document("", "x") == "title: none | text: x"


def test_build_item_body_with_abstract():
    assert build_item_body("Khalil, H.", 2002, "Prentice Hall", "An abstract.") == \
        "Khalil, H. (2002). Prentice Hall. An abstract."


def test_build_item_body_fallback_text_truncated_and_collapsed():
    from zotgemma import config

    body = build_item_body("", None, "", "", "foo \n\n  bar" + " x" * 10000)
    assert body.startswith("foo bar x")
    assert len(body) <= config.FALLBACK_TEXT_CHARS


def test_item_document():
    it = Item(1, "K", "book", "T", 1999, "A, B.", "V", "abs", "", "", "")
    assert item_document(it) == "title: T | text: A, B. (1999). V. abs"


# --- creators / years ------------------------------------------------------
def test_format_creator():
    assert format_creator("Brockett", "Roger W.", 0) == "Brockett, R. W."
    assert format_creator("Lions", "Jacques-Louis", 0) == "Lions, J. L."
    assert format_creator("UNESCO", "", 1) == "UNESCO"
    assert format_creator("Plato", None, 0) == "Plato"


def test_parse_year():
    assert parse_year("2004-00-00 2004") == 2004
    assert parse_year("March 5, 1998") == 1998
    assert parse_year("") is None and parse_year(None) is None


def test_parse_year_range():
    assert parse_year_range("2018:") == (2018, None)
    assert parse_year_range(":2005") == (None, 2005)
    assert parse_year_range("2010:2020") == (2010, 2020)
    assert parse_year_range("2012") == (2012, 2012)
    assert parse_year_range(None) == (None, None)


# --- eval / vectors --------------------------------------------------------
def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], ["b", "z"], 2) == (0.5, True)
    assert recall_at_k(["a", "b", "c"], ["z"], 3) == (0.0, False)


def test_truncate_renormalizes():
    v = np.array([[0.6, 0.0, 0.8, 0.0]], dtype=np.float32)
    t = truncate(v, 2)
    assert t.shape == (1, 2) and abs(np.linalg.norm(t) - 1) < 1e-6
    assert truncate(v, 4) is v


# --- sync / front matter ---------------------------------------------------
def _item(mod="2020", abstract="a"):
    return Item(1, "K", "book", "T", 2000, "", "", abstract, "", "", mod)


def test_is_unchanged_requires_trusted_hash():
    from zotgemma.index import is_unchanged

    old = {"date_modified": "2020", "fulltext_mtime": 1.0, "doc_text_hash": "h"}
    assert is_unchanged(old, _item(), 1.0, True, False)
    # interrupted sync left a NULL hash: must not be treated as unchanged
    assert not is_unchanged({**old, "doc_text_hash": None}, _item(), 1.0, True, False)
    assert not is_unchanged(old, _item(), 1.0, False, False)  # no vector
    assert not is_unchanged(old, _item(mod="2021"), 1.0, True, False)
    assert not is_unchanged(old, _item(), 1.0, True, True)  # force
    assert not is_unchanged(None, _item(), 1.0, True, False)
    assert not is_unchanged(old, _item(abstract=""), 2.0, True, False)  # text cache changed


def test_skip_front_matter():
    from zotgemma.index import skip_front_matter

    t = "Series editor blurb\nSpringer\n\nPreface\nThis book is about passivity.\n"
    assert skip_front_matter(t).startswith("Preface")
    t2 = "junk\nContents\n1 A\nAbstract\nReal abstract"
    assert skip_front_matter(t2).startswith("Abstract")  # abstract outranks contents
    assert skip_front_matter("no markers here") == "no markers here"
    assert skip_front_matter("1. Introduction\nbody").startswith("1. Introduction")


def test_best_attachment_prefers_largest_cache(tmp_path, monkeypatch):
    from zotgemma import config
    from zotgemma.models import Attachment
    from zotgemma.zotero_db import best_attachment

    monkeypatch.setattr(config, "STORAGE_DIR", tmp_path)
    for key, n in (("AAA", 5), ("BBB", 50)):
        (tmp_path / key).mkdir()
        (tmp_path / key / ".zotero-ft-cache").write_text("x" * n)
    a, b = Attachment(1, "AAA", 9, None, None), Attachment(2, "BBB", 9, None, None)
    assert best_attachment([a, b]) is b
    assert best_attachment([]) is None


def test_refresh_citekeys_cache_extra_and_live(monkeypatch):
    import sqlite3

    from zotgemma import citekeys, index

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE items (library_id INT, key TEXT, citekey TEXT)")
    conn.executemany("INSERT INTO items VALUES (1, ?, ?)", [("A", "cachedA"), ("B", None), ("C", "cachedC")])
    mk = lambda k, extra="": Item(1, k, "book", "", None, "", "", "", "", "", "", extra_citekey=extra)  # noqa: E731
    items = [mk("A"), mk("B", "extraB"), mk("C", "extraC")]
    # Zotero down: cached keys kept; extra only fills rows that have none
    monkeypatch.setattr(citekeys, "get_citekeys", lambda refs: ({}, citekeys.CACHED_SOURCE))
    st = index.SyncStats()
    index._refresh_citekeys(conn, items, st)
    assert dict(conn.execute("SELECT key, citekey FROM items")) == {"A": "cachedA", "B": "extraB", "C": "cachedC"}
    assert st.citekey_source == "cached (Zotero not running)" and st.citekeys_from_extra == 1
    # Zotero up: BBT wins; extra is the secondary source for items BBT has no key for
    monkeypatch.setattr(citekeys, "get_citekeys", lambda refs: ({(1, "A"): "liveA"}, "bbt"))
    index._refresh_citekeys(conn, items, index.SyncStats())
    assert dict(conn.execute("SELECT key, citekey FROM items")) == {"A": "liveA", "B": "extraB", "C": "extraC"}


def test_skip_front_matter_inline_abstract_beats_later_headings():
    from zotgemma.index import skip_front_matter

    t = "Title\n\x0cAbstract This monograph presents...\nmore\n1 Introduction\nbody\nContents\n"
    assert skip_front_matter(t).startswith("Abstract This monograph")
    assert skip_front_matter("x\nABSTRACT: we show\n").startswith("ABSTRACT:")
    assert skip_front_matter("x\nAbstract—we show\n").startswith("Abstract—")


def test_skip_front_matter_contents_regex_is_anchored():
    from zotgemma.index import skip_front_matter

    t = "Journal X\nContents lists available at ScienceDirect\nstuff\nSee the table of contents for details\n"
    assert skip_front_matter(t) == t  # neither line is a bare heading
    assert skip_front_matter("a\nTable of Contents\nb").startswith("Table of Contents")


def test_needs_recheck_on_doc_format_change_only_for_abstractless():
    """A doc_format bump re-hashes abstract-less items but leaves abstract items alone."""
    from zotgemma.index import needs_recheck

    old = {"date_modified": "2020", "fulltext_mtime": 1.0, "doc_text_hash": "h"}
    assert not needs_recheck(old, _item(abstract="a"), 1.0, True, False, True)
    assert needs_recheck(old, _item(abstract=""), 1.0, True, False, True)
    assert not needs_recheck(old, _item(abstract=""), 1.0, True, False, False)
    assert needs_recheck(old, _item(mod="2021"), 1.0, True, False, False)  # ordinary change still counts


def test_skip_front_matter_abstract_window_and_contents_lines():
    """Deep "abstract" lines (chapter abstracts, captions, contents entries) must not win over a preface."""
    from zotgemma import config
    from zotgemma.index import skip_front_matter

    filler = "lorem ipsum\n" * (config.ABSTRACT_SEARCH_CHARS // 12 + 10)
    book = "Series blurb\nPreface\nWhy this book.\n" + filler + "Abstract This chapter covers...\n"
    assert skip_front_matter(book).startswith("Preface")
    assert skip_front_matter("x\nabstract features\nHanddesigned program\n") == "x\nabstract features\nHanddesigned program\n"
    assert skip_front_matter("x\nAbstract Factory ** . . . 525\nContents\n").startswith("Contents")
    assert skip_front_matter("x\nAbstract Factory (99)\nIntroduction\n").startswith("Introduction") is False  # capitalised continuation is accepted early on
    assert skip_front_matter("x\nAbstract\nWe show...\n").startswith("Abstract\nWe show")


# --- Phase 1.5: discovery, index location, device, extra field -------------------------------
def _profile(home: Path, platform: str, data_dir: str | None, ini_body: str | None = None) -> Path:
    ini = discovery.profiles_ini_path(home, platform)
    prof = ini.parent / "Profiles" / "abc.default"
    prof.mkdir(parents=True)
    ini.write_text(ini_body or "[General]\n\n[Profile0]\nName=default\nIsRelative=1\nPath=Profiles/abc.default\nDefault=1\n")
    lines = ['user_pref("extensions.zotero.other", 1);']
    if data_dir:
        lines.append(f'user_pref("extensions.zotero.dataDir", "{data_dir}");')
    (prof / "prefs.js").write_text("\n".join(lines))
    return ini


def _zdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    (p / "zotero.sqlite").write_bytes(b"")
    return p


@pytest.mark.parametrize("platform,rel", [("darwin", "Library/Application Support/Zotero/profiles.ini"),
                                          ("linux", ".zotero/zotero/profiles.ini")])
def test_profiles_ini_paths(tmp_path, platform, rel):
    assert discovery.profiles_ini_path(tmp_path, platform) == tmp_path / rel


def test_discovery_order_flag_env_prefs_default(tmp_path):
    custom, flag, envd = _zdir(tmp_path / "custom"), _zdir(tmp_path / "flag"), _zdir(tmp_path / "env")
    _zdir(tmp_path / "Zotero")
    _profile(tmp_path, "linux", str(custom))
    d = lambda **kw: discovery.discover_zotero_dir(home=tmp_path, platform="linux", **kw)  # noqa: E731
    assert d(flag=flag, env={"ZOTGEMMA_ZOTERO_DIR": str(envd)}).path == flag
    assert d(env={"ZOTGEMMA_ZOTERO_DIR": str(envd)}).path == envd
    got = d(env={})
    assert got.path == custom and "extensions.zotero.dataDir" in got.source


def test_discovery_falls_back_to_default_and_lists_places(tmp_path):
    _profile(tmp_path, "linux", None)
    with pytest.raises(discovery.DiscoveryError) as e:
        discovery.discover_zotero_dir(home=tmp_path, platform="linux", env={})
    assert "prefs.js" in str(e.value) and str(tmp_path / "Zotero") in str(e.value)
    _zdir(tmp_path / "Zotero")
    assert discovery.discover_zotero_dir(home=tmp_path, platform="linux", env={}).path == tmp_path / "Zotero"


def test_discovery_explicit_source_is_final(tmp_path):
    _zdir(tmp_path / "Zotero")
    with pytest.raises(discovery.DiscoveryError, match="ZOTGEMMA_ZOTERO_DIR is .*nope"):
        discovery.discover_zotero_dir(home=tmp_path, platform="linux", env={"ZOTGEMMA_ZOTERO_DIR": str(tmp_path / "nope")})


def test_default_profile_selection(tmp_path):
    ini = tmp_path / "profiles.ini"
    ini.write_text("[Profile0]\nName=a\nIsRelative=1\nPath=Profiles/a\n\n[Profile1]\nName=b\nIsRelative=0\nPath=/abs/b\nDefault=1\n")
    assert discovery.default_profile_dir(ini) == Path("/abs/b")
    ini.write_text("[Profile0]\nName=a\nIsRelative=1\nPath=Profiles/a\n\n[Profile1]\nName=b\nPath=Profiles/b\n")
    assert discovery.default_profile_dir(ini) is None  # two profiles, none marked default
    ini.write_text("[Profile0]\nName=a\nIsRelative=1\nPath=Profiles/a\n")
    assert discovery.default_profile_dir(ini) == tmp_path / "Profiles/a"


def test_parse_prefs_unescapes_and_ignores_others():
    text = (r'user_pref("extensions.zotero.dataDir", "C:\\Users\\x\\Zotero");' + "\n"
            'user_pref("extensions.zotero.useDataDir", true);\nuser_pref("browser.x", "no");')
    p = discovery.parse_prefs(text)
    assert p == {"extensions.zotero.dataDir": "C:\\Users\\x\\Zotero", "extensions.zotero.useDataDir": "true"}


def test_index_path_is_per_library_and_stable(tmp_path):
    a, b = tmp_path / "A" / "Zotero", tmp_path / "B" / "Zotero"
    a.mkdir(parents=True); b.mkdir(parents=True)
    pa, pb = config.index_path_for(a, tmp_path / "data"), config.index_path_for(b, tmp_path / "data")
    assert pa != pb and pa == config.index_path_for(a, tmp_path / "data")
    assert pa.name == "index.sqlite" and pa.parent.parent == tmp_path / "data"


def test_batch_size_by_device_and_override():
    assert [config.default_batch_size(d, {}) for d in ("mps", "cuda", "cpu")] == [16, 64, 8]
    assert config.default_batch_size("mps", {"ZOTGEMMA_BATCH_SIZE": "5"}) == 5
    with pytest.raises(ValueError):
        config.default_batch_size("mps", {"ZOTGEMMA_BATCH_SIZE": "0"})


def test_pick_device():
    assert pick_device(None, True, True) == "mps"
    assert pick_device(None, False, True) == "cuda"
    assert pick_device(None, False, False) == "cpu"
    assert pick_device("cpu", True, True) == "cpu"
    with pytest.raises(RuntimeError):
        pick_device("cuda", True, False)


def test_extra_citekey_parser():
    assert parse_extra_citekey("Citation Key: smith2004\nfoo") == "smith2004"
    assert parse_extra_citekey("note\n  citation key:   a_b-1  ") == "a_b-1"
    assert parse_extra_citekey("Citation Keys: no") == "" and parse_extra_citekey(None) == ""


def test_resolve_attachment_paths(tmp_path):
    name, path = resolve_attachment_path("K", "attachments:papers/a.pdf", tmp_path)
    assert name == "a.pdf" and path == str(tmp_path / "papers/a.pdf")
    assert resolve_attachment_path("K", "/abs/x.pdf") == ("x.pdf", "/abs/x.pdf")
    assert resolve_attachment_path("K", None) == (None, None)
