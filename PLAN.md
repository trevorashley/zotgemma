# zotgemma: semantic + keyword search over the local Zotero library

Status: plan (2026-10-07). Project lives in `~/projects/zotgemma`.
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
~/projects/zotgemma/
  pyproject.toml            uv project; console scripts: zotgemma, zotgemma-mcp
  PLAN.md
  (index)                   <platformdirs user data>/zotgemma/<name>-<hash>/index.sqlite; never writes to the Zotero dir
  zotgemma/
    config.py               configure(): resolved paths, device, batch size; MODEL_ID, EMBED_DIM
    discovery.py            Zotero data-dir discovery (flag, env, profiles.ini/prefs.js, ~/Zotero)
    zotero_db.py            read-only snapshot of zotero.sqlite -> dataclasses
                            (items, creators, fields, tags, collections,
                             attachments, annotations, deletedItems)
    fulltext.py             .zotero-ft-cache loader; BM25 query on fulltext.sqlite
    citekeys.py             BBT JSON-RPC (libraryID:key), cached in index
    embedder.py             EmbeddingGemma 2 wrapper: lazy load, MPS, bf16/fp32,
                            query vs document formatting, truncate_dim, batching
    index.py                schema, upsert, incremental sync, deletion
    chunker.py              (phase 2) paragraph-aware token chunking
    search.py               dense + BM25 + reciprocal rank fusion, filters,
                            chunk->item rollup
    cli.py                  zotgemma index|search|show|status|eval
    mcp_server.py           (phase 3) stdio MCP server exposing search.py
  tests/
    golden.yaml             query -> expected item keys, for `zotgemma eval`
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
  query-time experiment via `zotgemma eval`, not a schema decision.
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
   Check: `uv run zotgemma --help` works.

2. **zotero_db.py.** Snapshot copy + queries:
   - bibliographic items (exclude attachment/note/annotation types, exclude
     deletedItems), with title, abstractNote, date (year), publicationTitle, DOI,
     url, itemType, dateModified;
   - creators joined in order (`itemCreators` + `creators`), formatted "Last, F.";
   - tags per item; collections per item (with full path names);
   - best PDF attachment per item (parentItemID, contentType = application/pdf,
     linkMode in {0,1}; path `storage:<file>` -> `~/Zotero/storage/<KEY>/<file>`);
   - the 47 standalone PDFs (no parent) as their own items, titled from filename.
   Check: `zotgemma status` prints counts matching the numbers above.

3. **fulltext.py.** `load_text(attachment_key)` reads `.zotero-ft-cache`;
   `bm25(query, limit)` runs `MATCH` against `fulltext.sqlite` (immutable) and maps
   rowid -> attachment -> parent item. Sanitize the query for FTS5 syntax (quote
   terms, strip operators) so natural-language queries do not error.
   Check: `zotgemma search --mode keyword "lyapunov"` returns ranked items.

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
   Check: second `zotgemma index` run embeds 0 items and finishes in seconds.

8. **search.py.** `search(query, limit=10, mode='hybrid', year_from/to=None,
   item_type=None, collection=None)`. Dense: top-100 from vec_items. Keyword:
   top-100 from `item_fts` BM25 plus Zotero full-text BM25 rolled to items. Fuse
   with RRF, apply filters, return top-k with per-source ranks for transparency.

9. **cli.py.** `zotgemma index`, `zotgemma search "query" [--mode] [--limit]
   [--year 2018:] [--type journalArticle] [--json]`, `zotgemma show <citekey|key>`
   (metadata, abstract, collections, tags, attachment path, annotations),
   `zotgemma status`, `zotgemma eval`.
   Output: rich table; `--json` for scripting; each row has the zotero:// link.

10. **eval.** Write `tests/golden.yaml` with 15–20 queries you know the answers
    to (concept-level, author-name, acronym, equation-name). `zotgemma eval`
    reports recall@5 / recall@10 for dense, keyword, hybrid, and for
    truncate_dim in {768, 512, 256}. This decides the defaults, not guesswork.

Phase 1 exit: hybrid recall@10 on the golden set is clearly better than keyword
alone, incremental sync works, and you use it for a week. (Done 2026-10-07.)

---

## Phase 1.5: run on any machine, against any Zotero library

Goal: `uv tool install` on a second computer, run `zotgemma index`, and it finds and indexes that
machine's Zotero library with no configuration.

Assumptions (agreed 2026-10-08): Zotero 10 or newer (so `fulltext.sqlite` with FTS5 exists);
Better BibTeX installed; `google/embeddinggemma-2` already in the Hugging Face cache; Linux with
CUDA or macOS with Metal. Windows and Zotero 6/7 are out of scope.

1. **Library discovery.** Resolve the data directory in this order: `--zotero-dir` flag,
   `ZOTGEMMA_ZOTERO_DIR`, then Zotero's own preference `extensions.zotero.dataDir` read from
   `prefs.js` of the default profile (`~/Library/Application Support/Zotero/profiles.ini` on
   macOS, `~/.zotero/zotero/profiles.ini` on Linux), then `~/Zotero`. Fail with a message naming
   each place looked. Refuse data dirs whose `zotero.sqlite` schema version predates Zotero 10
   (check `version` table, `userdata`), naming the version found.

2. **Index location.** Store the index under the user's data dir via `platformdirs`
   (`~/Library/Application Support/zotgemma/` or `~/.local/share/zotgemma/`), one subdirectory per
   library keyed by a hash of the data-dir path, so one install serves several libraries.
   `--index` / `ZOTGEMMA_INDEX_DB` still override. `zotgemma status` prints both paths.

3. **Group libraries.** Load items from every library in `libraries`, carry `library_id` in
   `items` and `attachments`, make `(library_id, key)` the unique key, and build links as
   `zotero://select/library/items/KEY` for the user library and
   `zotero://select/groups/<groupID>/items/KEY` otherwise (`groups.libraryID -> groupID`).
   Verify what the BBT RPC `item.citationkey` accepts for group items (plain key vs
   `libraryID:key`) and pass that. Add `--library` filter to `search`.

4. **Any attachment with a text cache.** Index HTML snapshots and EPUBs alongside PDFs: select
   attachments by the presence of `storage/<KEY>/.zotero-ft-cache`, not by `contentType`.
   Linked files (`linkMode` 2) keep their text cache in storage, so only the displayed path
   changes; resolve `attachments:` prefixes against `extensions.zotero.baseAttachmentPath`.

5. **Cite keys with BBT only.** Drop the `better-bibtex.migrated` fallback. When Zotero is not
   running, keep the cached keys and report "citekeys: cached (Zotero not running)". As a
   secondary source for items BBT does not return, parse a `Citation Key:` line from the `extra`
   field (BBT writes it there for pinned keys).

6. **Device and batch size.** Keep the existing MPS/CUDA/CPU selection and bf16 check. Add
   `--device` and `ZOTGEMMA_DEVICE` overrides, and pick the embedding batch size by device
   (16 on MPS, 64 on CUDA, 8 on CPU) with `ZOTGEMMA_BATCH_SIZE` to override. Honour `HF_HOME`
   and `HF_HUB_OFFLINE`; the model is loaded with `local_files_only=True` first, as now.

7. **Packaging.** Relax `requires-python` to `>=3.11` and confirm the lock resolves on 3.11 to
   3.14 for linux-x86_64 and macos-arm64. Document that sqlite-vec needs a Python with extension
   loading (uv-managed and Homebrew Pythons have it; the macOS system Python does not) and that
   `uv tool install git+<repo>` is the supported install. Add a `--version` flag.

8. **Tests without a real library.** Build a tiny synthetic Zotero 10 data dir fixture (a
   `zotero.sqlite` with the handful of tables we read, two items, one group item, one PDF and one
   HTML attachment with text caches, a `fulltext.sqlite` with one FTS5 row) and run
   `load_items`, `sync` with a stubbed embedder, and `search --mode keyword` against it in CI.
   Make the golden set optional: `zotgemma eval --golden PATH`, with `tests/golden.yaml` kept as
   the example for this library.

Phase 1.5 exit: a clean clone on a second machine indexes its own library with `zotgemma index`
and no flags; the synthetic-fixture tests pass without Zotero installed.

**Status: implemented 2026-10-08.** Notes and deviations:

- Item 1: `userdata` is 130 on Zotero 10.0.5. The threshold is 125 (refuse below): the first number used by
  10.0.0 is unknown, so a low bound avoids rejecting a valid early 10.x; the functional requirement
  (`fulltext.sqlite` with FTS5) fails loudly on its own. An explicit `--zotero-dir`/`ZOTGEMMA_ZOTERO_DIR`
  that lacks `zotero.sqlite` is a hard error rather than falling through. `dataDir` is used only when `useDataDir` is true (Zotero's default is false and prefs.js omits defaults, so a bare dataDir line is stale).
  Options live on the top-level command (`zotgemma --zotero-dir X status`); `--device` is also accepted by
  `index`, `search` and `eval`.
- Item 2: on first use the legacy `data/index.sqlite` is copied to the new location (only when the library is
  the one the old code would have used, `ZOTGEMMA_ZOTERO_DIR` or `~/Zotero`); the old file is left in place.
  The index schema is migrated in place (`items` rebuilt with `library_id`, `group_id`, `library_name` and
  `UNIQUE(library_id, key)`; `attachments` gains `library_id`, `content_type`) without touching vectors.
- Item 3: BBT `item.citationkey` takes `[libraryID]:[itemKey]` (checked in the BBT source and live:
  `JKLLQ43U` and `1:JKLLQ43U` both return the key, `2:JKLLQ43U` returns null). The number is the Zotero
  libraryID, not the groupID; a bare key means the user library only. zotgemma always sends `libraryID:key`.
  Group handling is covered by the synthetic fixture only; this library has no groups. `--library` accepts
  `user`, a groupID or a group-name substring.
- Item 4: attachments are kept if they have a `.zotero-ft-cache`, or are imported PDFs (so missing text is
  still reported). The "best" attachment prefers PDFs, then the largest cache, so existing document strings
  did not change. On this library the first sync after the change re-embedded 5 abstract-less items whose
  only attachment is an HTML snapshot (6 s); the following run embedded 0 in 0.4 s.
- Item 5: `better-bibtex.migrated` support removed. Offline sync keeps cached keys; `Citation Key:` from
  Extra fills rows with none (offline) or items BBT returns nothing for (online).
- Item 7: lock resolves universally for 3.11 to 3.14. `uv pip compile` succeeds for linux x86_64 on 3.11 and
  3.14; for aarch64-apple-darwin it fails only because uv assumes macOS 13 and torchvision 0.29.1 ships
  `macosx_14_0_arm64` wheels. A real `uv sync --locked --python 3.11` on macOS 14+ arm64 installed and passed
  the test suite.
- Legacy index adoption additionally requires a sample of the legacy index's item keys to exist in the current
  `zotero.sqlite`. Tests are hermetic (autouse fixture redirects HOME, the index path and the legacy path).
- Known limitations: `baseAttachmentPath` is read from the default profile only; feed libraries are not
  excluded from the `status` item counts; `show` picks the lowest libraryID when a key exists in several
  libraries; a bad `--device`/`ZOTGEMMA_DEVICE` makes `status` fail too (it is validated at startup).
- Item 8: fixture in `tests/conftest.py`; CI-free, runs with `uv run pytest`.

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
   time (likely several hours on M1). `zotgemma index --chunks` must be resumable
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

5. **CLI.** `zotgemma passages "query"`, `zotgemma search --with-passages` (shows
   best snippet under each item), `zotgemma show <key> --text [--grep term]`.

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
   indexing from MCP; `zotgemma index` stays a CLI/cron job (launchd every night,
   or on demand).

3. **Registration.**
   `claude mcp add --scope user zotero -- uv run --directory ~/projects/zotgemma zotgemma-mcp`
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

## Phase 4: topics (the inverse problem: cluster the library into auto-collections)

Goal: `zotgemma topics` groups the whole library into a two-level tree of named topics, shows how
each topic lines up with the user's Zotero collections, and points out uncollected or possibly
misfiled items. Like collections, but derived from the content.

Experiment (2026-10-08, item vectors from Phase 1, scikit-learn, ~1 s): k-means and Ward linkage at
k=20 both gave balanced, readable topics (root locus; safe RL; consensus; Kalman/Bayesian filtering;
temporal logic; synthetic aperture sonar; differential geometry; geometric mechanics; convex and
Riemannian optimization; control education; C++ design patterns; MPC; ...). Adjusted mutual
information against the top-level collection was 0.33, which is expected: "Reviews" is a
document-type collection and spreads over a dozen topics. Average-linkage agglomerative failed
(one 973-item cluster plus singletons); do not use it. Abstract-less books titled from filenames
formed junk clusters keyed on "pdf", so Phase 2 (chunk vectors) should land first.

Depends on Phase 1 only; benefits from Phase 2. Independent of Phase 3.

1. **Clustering core (`topics.py`).** Load all item vectors from `vec_items`. Ward linkage on the
   unit vectors (scikit-learn `AgglomerativeClustering`, `linkage="ward"`), cut at two levels:
   parents (~6-10) and children (~30-50). Choose each cut by silhouette score over a range rather
   than a fixed k; `--k` and `--parents` override. Record per item: parent, child, distance to its
   centroid and margin to the second-nearest centroid (small margin = "ambiguous"). Persist in a
   `topics` table (topic_id, parent_id, level, name, terms, size, centroid blob) and an
   `item_topics` table (item_id, topic_id, distance, margin) in the index DB, with a `meta` key for
   the clustering run id. Add `scikit-learn` as a dependency.

2. **Naming.** Deterministic labels first: class-based TF-IDF (1-2 grams, English stop words,
   `max_df` 0.3) over title + abstract, top 6 terms per topic. Optional `--name-with ollama:<model>`
   or `--name-with claude` sends the top ten titles of each topic and asks for a 2-4 word name;
   store both, display the generated name with the terms underneath. Never block on a naming
   backend: fall back to the terms.

3. **Comparison with collections.** For each topic: the best-matching Zotero collection (by
   overlap), its purity, and the count of items with no collection. Two reports:
   `topics --uncollected` lists items in no collection grouped by topic (a ready-made filing list);
   `topics --misfiled` lists items whose topic's dominant collection differs from their own, sorted
   by centroid distance (most confident first). Both are read-only.

4. **CLI and output.** `zotgemma topics` prints the tree (name, size, terms, matching collection,
   three medoid titles). `--json` for scripting; `--items <topic>` lists a topic's members with
   citekeys and links. `--map out.html` writes a self-contained 2-D map (PCA or UMAP if installed,
   never a hard dependency) with one point per item, coloured by parent topic, hover showing title
   and citekey, click opening the `zotero://select` link.

5. **Stability across syncs.** `zotgemma index` assigns new items to the nearest existing
   centroid (child, then parent) so topics stay usable between full runs. `zotgemma topics
   --recluster` recomputes everything and matches new topics to old ones by centroid similarity
   (Hungarian assignment) so ids and generated names carry over where the topic persisted.

6. **Experiments to run and record in Benchmarks.** (a) Re-embed with EmbeddingGemma 2's
   clustering prompt (`task: clustering | query: ...`, ~10 min) into a separate `vec_items_cluster`
   table and compare silhouette and collection AMI against the retrieval vectors; keep whichever
   wins. (b) After Phase 2: soft membership, an item belongs to every topic that holds at least a
   configurable share of its chunks, so books spanning subjects appear under each.

7. **Writing back to Zotero (later, opt-in).** SQLite is never written. If wanted, `topics --push`
   creates collections through the Zotero Web API (pyzotero, needs an API key with write access)
   under one parent collection named "Auto topics", adds items by key, and never modifies existing
   collections. Requires explicit confirmation on every run and a dry-run listing first. Out of
   scope until the read-only reports have been used for a while.

Phase 4 exit: `zotgemma topics` produces a tree you would accept as a first draft of collections,
the uncollected report files the items currently in no collection, and the Phase 2 chunk vectors
have removed the filename-titled junk clusters.

---

## Risks and notes

- MPS bf16: works in recent torch but verify numerically on first load (step 1.5).
- First run downloads the full model (~1.5 GB incl. vision/audio encoders) to the
  HF cache; text-only selective loading is a nice-to-have, not a blocker.
- Zotero's FTS has indexed 1,328 of 1,364 text caches; a handful of PDFs have no
  extracted text (scans). OCR is out of scope; `zotgemma status` should list them.
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
