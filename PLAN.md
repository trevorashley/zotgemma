# zotero-search: semantic + keyword search over the local Zotero library

Status: plan (2026-10-07). Project lives in `~/projects/zotero-search`.
The earlier `~/projects/zotero` (pyzotero + Ollama keyword experiment, conda py3.10) is left untouched.

## Facts the design rests on (verified 2026-10-07)

- Zotero data dir: `~/Zotero`. Zotero 7 is running, so `zotero.sqlite` is locked.
  Read it by copying to a temp file (21 MB) or opening with `?immutable=1`.
- Library: ~1,200 bibliographic items (663 journal articles, 316 books, 82 conf papers,
  67 preprints), 1,317 PDF attachments, 499 highlight annotations (~84k chars),
  119 collections, 4,300 tags. All in libraryID 1 (no group libraries).
- Zotero has already extracted text: `storage/<KEY>/.zotero-ft-cache` (plain text,
  1,364 files, 146 MB total ≈ 35M tokens). No PDF parsing needed.
- `fulltext.sqlite` has an FTS5 table `fulltextContent` (contentless, unicode61).
  `rowid` == attachment `itemID`. Gives BM25 ranking for free; cannot return text
  (read `.zotero-ft-cache` for snippets).
- Better BibTeX cite keys: live via JSON-RPC
  `POST http://localhost:23119/better-bibtex/json-rpc` method `item.citationkey`
  (verified working). Offline fallback: `~/Zotero/better-bibtex.migrated` is a SQLite
  file with table `citationkey(itemID, itemKey, libraryID, citationKey, ...)`, 1,004 rows,
  but it is a Feb-2026 snapshot and may be stale.
- Model: `google/embeddinggemma-2`. Text backbone 270M params, 768-d output,
  Matryoshka truncation to 512/256/128, 8k context, Apache 2.0, no gating.
  bf16 or fp32 only (fp16 produces NaNs). sentence-transformers prompt names:
  `SearchQuery` for queries, `Document` for passages. The `Document` prompt is
  `title: none | text: ...`, so for our own titles build the string by hand:
  `f"title: {title} | text: {body}"` and encode with no prompt_name.
- Toolchain: Apple M1, 16 GB. Python 3.14.8 (mise) + uv. `uv pip compile` on 3.14 /
  arm64 resolves torch 2.11.0, sentence-transformers 6.1.0, transformers 5.19.0,
  sqlite-vec 0.1.9, mcp 2.3.0. No need to pin an older Python.
- Claude Code already has an MCP server configured (Wolfram) in `~/.claude.json`,
  so `claude mcp add` is the registration path. No Claude Desktop config exists yet.

## Architecture

```
~/projects/zotero-search/
  pyproject.toml            uv project; console scripts: zsearch, zsearch-mcp
  PLAN.md
  data/index.sqlite         our index (gitignored); never writes to ~/Zotero
  zsearch/
    config.py               ZOTERO_DIR, INDEX_DB, MODEL_ID, EMBED_DIM, paths
    zotero_db.py            read-only snapshot of zotero.sqlite -> dataclasses
                            (items, creators, fields, tags, collections,
                             attachments, annotations, deletedItems)
    fulltext.py             .zotero-ft-cache loader; BM25 query on fulltext.sqlite
    citekeys.py             BBT JSON-RPC with .migrated fallback, cached in index
    embedder.py             EmbeddingGemma 2 wrapper: lazy load, MPS, bf16/fp32,
                            query vs document formatting, truncate_dim, batching
    index.py                schema, upsert, incremental sync, deletion
    chunker.py              (phase 2) paragraph-aware token chunking
    search.py               dense + BM25 + reciprocal rank fusion, filters,
                            chunk->item rollup
    cli.py                  zsearch index|search|show|status|eval
    mcp_server.py           (phase 3) stdio MCP server exposing search.py
  tests/
    golden.yaml             query -> expected item keys, for `zsearch eval`
```

Index DB schema (SQLite + sqlite-vec):

```
items      (item_id PK, key, citekey, item_type, title, year, authors, venue,
            abstract, doi, url, date_modified, fulltext_mtime, doc_text_hash,
            attachment_key, attachment_path)
item_fts   FTS5(title, authors, abstract, tags)    -- our own BM25 over metadata
vec_items  vec0(item_id, embedding float[768])
chunks     (chunk_id PK, item_id, attachment_id, seq, kind{'text','annotation'},
            char_start, char_end, text)                          -- phase 2
vec_chunks vec0(chunk_id, embedding float[768])                  -- phase 2
meta       (k, v)   -- model id, dim, schema version, last sync
```

Design rules:
- Nothing ever writes to `~/Zotero`. Snapshot `zotero.sqlite` to a temp copy on
  each sync (cheap, avoids WAL/lock edge cases).
- Dense vectors stored at full 768-d, normalized. Truncation to 256/512 is a
  query-time experiment via `zsearch eval`, not a schema decision.
- Hybrid by default: reciprocal rank fusion (k=60) over dense and BM25 lists.
  Pure modes stay available for debugging (`--mode dense|keyword|hybrid`).
- Hybrid fusion is two-stage: the metadata-BM25 and Zotero full-text-BM25 lists are RRF-fused into one
  keyword list, which is then RRF-fused (k=60) with the dense list. A flat 3-way RRF let the two keyword
  lists outvote dense and scored worse (hit@10 0.789 vs 0.842 on the first 19-query golden set).
- Every PDF of an item is tracked in the `attachments` table; the "best" one (largest text cache) feeds
  the abstract-less fallback text, while full-text BM25 and annotations cover all of them.
- Every result carries: citekey, title, authors, year, score, source of hit,
  and a `zotero://select/library/items/<KEY>` link.

---

## Phase 1: item-level index, hybrid search, CLI

Goal: "which papers in my library are about X" in under a second, from the shell.

1. **Scaffold.** `uv init --package`, Python 3.14, deps: torch, sentence-transformers,
   sqlite-vec, numpy, typer, rich, httpx (BBT RPC), pyyaml. `.gitignore` data/ and
   the HF cache. `git init`.
   Check: `uv run zsearch --help` works.

2. **zotero_db.py.** Snapshot copy + queries:
   - bibliographic items (exclude attachment/note/annotation types, exclude
     deletedItems), with title, abstractNote, date (year), publicationTitle, DOI,
     url, itemType, dateModified;
   - creators joined in order (`itemCreators` + `creators`), formatted "Last, F.";
   - tags per item; collections per item (with full path names);
   - best PDF attachment per item (parentItemID, contentType = application/pdf,
     linkMode in {0,1}; path `storage:<file>` -> `~/Zotero/storage/<KEY>/<file>`);
   - the 47 standalone PDFs (no parent) as their own items, titled from filename.
   Check: `zsearch status` prints counts matching the numbers above.

3. **fulltext.py.** `load_text(attachment_key)` reads `.zotero-ft-cache`;
   `bm25(query, limit)` runs `MATCH` against `fulltext.sqlite` (immutable) and maps
   rowid -> attachment -> parent item. Sanitize the query for FTS5 syntax (quote
   terms, strip operators) so natural-language queries do not error.
   Check: `zsearch search --mode keyword "lyapunov"` returns ranked items.

4. **citekeys.py.** Batch `item.citationkey` over all item keys via JSON-RPC;
   if Zotero is not running, fall back to `better-bibtex.migrated`; cache in
   `items.citekey`.

5. **embedder.py.** Load once per process. Device = mps if available. Try bf16 on
   MPS first; if outputs contain NaN or similarity sanity check fails, fall back
   to fp32. Expose `embed_documents(list[(title, body)])` and
   `embed_query(str)`; batch size ~16; normalize. Benchmark tokens/sec on 200
   docs and record it in PLAN.md (needed to size phase 2).
   Check: known-similar pair scores > known-unrelated pair.

6. **Document text per item.** `title: {title} | text: {authors} ({year}). {venue}.
   {abstract}`. If no abstract, substitute the first ~1,500 tokens of the
   `.zotero-ft-cache` (skipping front matter is a later refinement). Hash the
   string; re-embed only when the hash changes.

7. **index.py.** Create schema; `sync()` = snapshot -> diff by (dateModified,
   fulltext mtime, doc_text_hash) -> embed changed -> upsert -> delete items no
   longer present. Also populate `item_fts`.
   Check: second `zsearch index` run embeds 0 items and finishes in seconds.

8. **search.py.** `search(query, limit=10, mode='hybrid', year_from/to=None,
   item_type=None, collection=None)`. Dense: top-100 from vec_items. Keyword:
   top-100 from `item_fts` BM25 plus Zotero full-text BM25 rolled to items. Fuse
   with RRF, apply filters, return top-k with per-source ranks for transparency.

9. **cli.py.** `zsearch index`, `zsearch search "query" [--mode] [--limit]
   [--year 2018:] [--type journalArticle] [--json]`, `zsearch show <citekey|key>`
   (metadata, abstract, collections, tags, attachment path, annotations),
   `zsearch status`, `zsearch eval`.
   Output: rich table; `--json` for scripting; each row has the zotero:// link.

10. **eval.** Write `tests/golden.yaml` with 15–20 queries you know the answers
    to (concept-level, author-name, acronym, equation-name). `zsearch eval`
    reports recall@5 / recall@10 for dense, keyword, hybrid, and for
    truncate_dim in {768, 512, 256}. This decides the defaults, not guesswork.

Phase 1 exit: hybrid recall@10 on the golden set is clearly better than keyword
alone, incremental sync works, and you use it for a week.

---

## Phase 2: full-text chunks and annotations

Goal: "where did I read about X" — passage-level hits with page/offset, rolled up
to items.

1. **chunker.py.** Use the model tokenizer. Split `.zotero-ft-cache` on blank
   lines, pack paragraphs to ~700 tokens, overlap ~80 tokens, keep char offsets.
   Drop chunks that are mostly non-alphabetic (reference lists, tables of
   numbers) via a cheap ratio heuristic. Expected volume: ~45k chunks.
   Also one chunk per annotation (highlight text + comment), kind='annotation',
   linked to the parent item, with pageLabel kept for display.

2. **Throughput and scheduling.** From the phase-1 benchmark, estimate total
   time (likely several hours on M1). `zsearch index --chunks` must be resumable
   (commit per attachment, skip already-done attachments by fulltext mtime) and
   safe to Ctrl-C. Order: items by dateAdded desc, so recent papers are
   searchable first. Optional `--max-chunks-per-item N` to cap long books on the
   first pass.

3. **Schema and sync.** `chunks` + `vec_chunks` as above. Deleting an item deletes
   its chunks. A changed `.zotero-ft-cache` mtime re-chunks that attachment only.

4. **search.py extensions.** `search_passages(query, limit, item_key=None)`
   returns chunks with text, item, char range. Item search gains a third signal:
   max chunk score per item (rolled up) fused via RRF with the existing two.
   Annotation chunks get a configurable boost (they are text you already judged
   important).

5. **CLI.** `zsearch passages "query"`, `zsearch search --with-passages` (shows
   best snippet under each item), `zsearch show <key> --text [--grep term]`.

6. **Eval.** Extend golden.yaml with passage-level queries. Compare chunk sizes
   (500 / 700 / 1000) and the rollup weighting on recall@10.

Phase 2 exit: passage search finds specific claims/proofs you remember reading,
and the full chunk index is built and stays in sync.

---

## Phase 3: MCP server

Goal: Claude Code (and Claude Desktop later) can search your library by concept,
read passages, and cite with Better BibTeX keys.

1. **mcp_server.py** using the `mcp` SDK, stdio transport, FastMCP-style tools:
   - `search_library(query, limit=10, mode='hybrid', year_from, year_to,
     item_type, collection)` -> list of {citekey, key, title, authors, year,
     venue, score, best_snippet, zotero_link}
   - `search_passages(query, limit=10, item_key=None)` -> list of
     {citekey, title, page_hint, text, char_start, char_end}
   - `get_item(key_or_citekey)` -> full metadata, abstract, tags, collections,
     annotations, attachment path
   - `get_item_text(key_or_citekey, start=0, length=8000)` -> slice of full text
     (paged so a 400-page book cannot blow the context)
   - `list_collections()` -> tree with item counts
   - `index_status()` -> counts, last sync, model, whether Zotero is running
   Tool descriptions must say what the library is (your personal Zotero library,
   ~1,200 items in control theory / robotics / math / software) so the model
   reaches for it at the right moments.

2. **Process model.** Embedder loads lazily on first dense call (a few seconds);
   process stays alive for the session. Keyword-only tools answer instantly. No
   indexing from MCP; `zsearch index` stays a CLI/cron job (launchd every night,
   or on demand).

3. **Registration.**
   `claude mcp add --scope user zotero -- uv run --directory ~/projects/zotero-search zsearch-mcp`
   Then test from a Claude Code session: "find papers in my library about
   passivity-based control" and "what does Brockett say about nonholonomic
   stabilization" (passage-level).
   Claude Desktop: same command in `claude_desktop_config.json` if wanted.

4. **Optional reranker.** Only if eval shows top-10 is right but ordering is
   poor: add a cross-encoder (e.g. a small BGE reranker) over the fused top-30.
   Measure before adding; it costs latency on every call.

Phase 3 exit: from Claude Code, a concept question returns correct papers with
cite keys and quotable passages, without you opening Zotero.

---

## Risks and notes

- MPS bf16: works in recent torch but verify numerically on first load (step 1.5).
- First run downloads the full model (~1.5 GB incl. vision/audio encoders) to the
  HF cache; text-only selective loading is a nice-to-have, not a blocker.
- Zotero's FTS has indexed 1,328 of 1,364 text caches; a handful of PDFs have no
  extracted text (scans). OCR is out of scope; `zsearch status` should list them.
- `~/projects/zotero/test.py` contains a Zotero Web API key in plain text.
  Rotate it at zotero.org/settings/keys; this project never needs the web API.

---

## Benchmarks (measured 2026-10-07, Phase 1 build)

Machine: Apple M1, 16 GB, torch 2.14.1 on MPS. Model: `google/embeddinggemma-2`, text-only
(`config_kwargs={"vision_config": None, "audio_config": None}`, 271M params), bf16
(sanity check passed: similar pair 0.846 vs unrelated 0.479; no NaNs), batch size 16,
`max_seq_length` 1024.

- Full item index: 1,243 documents, 654,494 tokens (mean ~527 tokens/doc; abstract-less items use
  the first ~6,000 characters of the text cache), embedded in **513 s of encode time**
  (523.7 s wall-clock including model load and sync) = **2.42 docs/s, ~1,276 tokens/s**.
- No-op re-sync (nothing changed): 0.4 s of work, 0.63 s total process time.
- Phase 2 sizing: the plan's ~35M tokens of full text at ~1,276 tokens/s is about 27,400 s,
  roughly **7.6 hours** for a first full pass. ~700-token chunks attend over shorter sequences than
  the ~1,000-token documents here, so real throughput should be somewhat better; treat 5 to 8 hours as
  the range and keep `--max-chunks-per-item` and resumability in the design.
- Re-embeds after review fixes: `doc_format` 2 (fallback text ~3,600 chars starting at
  Abstract/Preface/Introduction/Contents) re-embedded 482 abstract-less items, 413,528 tokens in 334.5 s
  encode = 1.44 items/s, ~1,236 tokens/s; `doc_format` 3 (inline "Abstract ..." headings) re-embedded
  135 items in 92 s; `doc_format` 4 (an Abstract heading only counts within the first 10,000 chars, and
  caption/contents lines are rejected) re-embedded the items listed in the git log for that commit.
- Golden-set eval, 36 queries (10+10 concept, 7 equation, 4 author, 5 acronym), mean fraction of expected
  found (recall) / any-expected (hit):

  | config | recall@5 | recall@10 | hit@5 | hit@10 |
  |---|---|---|---|---|
  | keyword | 0.713 | 0.833 | 0.750 | 0.833 |
  | dense@768 | 0.706 | 0.832 | 0.778 | 0.861 |
  | dense@512 | 0.692 | 0.803 | 0.778 | 0.833 |
  | dense@256 | 0.694 | 0.725 | 0.778 | 0.778 |
  | hybrid@768 | 0.782 | 0.856 | 0.833 | 0.861 |
  | hybrid@512 | 0.782 | 0.836 | 0.833 | 0.861 |
  | hybrid@256 | 0.776 | 0.836 | 0.833 | 0.861 |

  Hybrid wins on recall@5/@10 and hit@5. Dense truncation to 512 now costs a little (0.832 to 0.803 recall@10; 256 drops to 0.725),
  so keep 768. The earlier 19-query set (superseded) gave hybrid 0.842 vs keyword 0.737 recall@10.
