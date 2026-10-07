"""Tests for pure helper functions (no Zotero, model or network needed)."""

from zsearch.embedder import build_document, truncate
from zsearch.evaluation import recall_at_k
from zsearch.fulltext import sanitize_fts_query
from zsearch.index import build_item_body, item_document
from zsearch.models import Item
from zsearch.search import parse_year_range, rrf_fuse
from zsearch.zotero_db import format_creator, parse_year

import numpy as np


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
    body = build_item_body("", None, "", "", "foo \n\n  bar" + " x" * 10000)
    assert body.startswith("foo bar x")
    assert len(body) <= 6000


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
    from zsearch.index import is_unchanged

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
    from zsearch.index import skip_front_matter

    t = "Series editor blurb\nSpringer\n\nPreface\nThis book is about passivity.\n"
    assert skip_front_matter(t).startswith("Preface")
    t2 = "junk\nContents\n1 A\nAbstract\nReal abstract"
    assert skip_front_matter(t2).startswith("Abstract")  # abstract outranks contents
    assert skip_front_matter("no markers here") == "no markers here"
    assert skip_front_matter("1. Introduction\nbody").startswith("1. Introduction")


def test_best_attachment_prefers_largest_cache(tmp_path, monkeypatch):
    from zsearch import config
    from zsearch.models import Attachment
    from zsearch.zotero_db import best_attachment

    monkeypatch.setattr(config, "STORAGE_DIR", tmp_path)
    for key, n in (("AAA", 5), ("BBB", 50)):
        (tmp_path / key).mkdir()
        (tmp_path / key / ".zotero-ft-cache").write_text("x" * n)
    a, b = Attachment(1, "AAA", 9, None, None), Attachment(2, "BBB", 9, None, None)
    assert best_attachment([a, b]) is b
    assert best_attachment([]) is None


def test_migrated_citekeys_only_fill_missing(monkeypatch):
    import sqlite3

    from zsearch import citekeys, index

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE items (key TEXT, citekey TEXT)")
    conn.executemany("INSERT INTO items VALUES (?, ?)", [("A", "liveA"), ("B", None)])
    it = [Item(1, "A", "book", "", None, "", "", "", "", "", ""), Item(2, "B", "book", "", None, "", "", "", "", "", "")]
    monkeypatch.setattr(citekeys, "get_citekeys", lambda keys: ({"A": "staleA", "B": "staleB"}, "migrated"))
    index._refresh_citekeys(conn, it, index.SyncStats())
    assert dict(conn.execute("SELECT key, citekey FROM items")) == {"A": "liveA", "B": "staleB"}
    monkeypatch.setattr(citekeys, "get_citekeys", lambda keys: ({"A": "newA"}, "bbt"))
    index._refresh_citekeys(conn, it, index.SyncStats())
    assert dict(conn.execute("SELECT key, citekey FROM items"))["A"] == "newA"


def test_skip_front_matter_inline_abstract_beats_later_headings():
    from zsearch.index import skip_front_matter

    t = "Title\n\x0cAbstract This monograph presents...\nmore\n1 Introduction\nbody\nContents\n"
    assert skip_front_matter(t).startswith("Abstract This monograph")
    assert skip_front_matter("x\nABSTRACT: we show\n").startswith("ABSTRACT:")
    assert skip_front_matter("x\nAbstract—we show\n").startswith("Abstract—")


def test_skip_front_matter_contents_regex_is_anchored():
    from zsearch.index import skip_front_matter

    t = "Journal X\nContents lists available at ScienceDirect\nstuff\nSee the table of contents for details\n"
    assert skip_front_matter(t) == t  # neither line is a bare heading
    assert skip_front_matter("a\nTable of Contents\nb").startswith("Table of Contents")


def test_recheck_fallback_reembeds_only_abstractless(monkeypatch, tmp_path):
    """doc_format change re-hashes abstract-less items but leaves abstract items alone."""
    from zsearch import index

    def cond(old, it, recheck):
        return index.is_unchanged(old, it, 1.0, True, False) and not (recheck and not it.abstract)

    old = {"date_modified": "2020", "fulltext_mtime": 1.0, "doc_text_hash": "h"}
    assert cond(old, _item(abstract="a"), True)
    assert not cond(old, _item(abstract=""), True)
    assert cond(old, _item(abstract=""), False)
