# zotgemma

A better Zotero search that leverages EmbeddingGemma 2: semantic + keyword search over a local Zotero library. Dense retrieval with
`google/embeddinggemma-2`, BM25 over item metadata and Zotero's own full-text index,
fused with reciprocal rank fusion (k=60). Read-only with respect to the Zotero data directory.
See `PLAN.md` for the design and phases.

## Install

Requires [uv](https://docs.astral.sh/uv/), Zotero 10 or newer (its `fulltext.sqlite` and extracted-text
cache are what gets indexed), the Better BibTeX plugin for cite keys, and Python 3.11 or newer.

```sh
uv tool install git+https://github.com/trevorashley/zotgemma
zotgemma --version
zotgemma index
```

From a clone: `uv sync && uv run zotgemma --help`.

Notes:

- `sqlite-vec` needs a Python whose `sqlite3` module can load extensions. uv-managed and Homebrew Pythons can;
  the macOS system Python (`/usr/bin/python3`) cannot. If you see "cannot load extensions", install with
  `uv tool install --python 3.12 git+https://github.com/trevorashley/zotgemma`.
- Linux with CUDA or macOS with Metal (MPS) is assumed; CPU works but is slow. Windows is untested.
- The first embedding run downloads `google/embeddinggemma-2` (~1.5 GB) into the Hugging Face cache. It honours
  `HF_HOME` and `HF_HUB_OFFLINE`; the local cache is tried first.

## Where things live

Zotero data directory, first match wins:

1. `--zotero-dir PATH` (global option, before the subcommand)
2. `ZOTGEMMA_ZOTERO_DIR`
3. `extensions.zotero.dataDir` from `prefs.js` of Zotero's default profile (`profiles.ini` at
   `~/Library/Application Support/Zotero/` on macOS, `~/.zotero/zotero/` on Linux)
4. `~/Zotero`

An explicit flag or variable that points nowhere is an error; otherwise the message lists every place looked.
`zotero.sqlite` must have a `userdata` schema version of at least 125 (Zotero 10.0.5 writes 130); older
databases are refused.

The index is stored per library under the platform user-data directory
(`~/Library/Application Support/zotgemma/<name>-<hash>/index.sqlite` on macOS,
`~/.local/share/zotgemma/<name>-<hash>/index.sqlite` on Linux), the hash being of the data-dir path, so one install
can serve several libraries. `--index PATH` / `ZOTGEMMA_INDEX_DB` override it. A pre-existing
`data/index.sqlite` from a source checkout is copied to the new location on first use (no re-embedding).
`zotgemma status` prints both paths.

## Index

```sh
zotgemma status        # data dir, index path, library + index counts, attachments by content type
zotgemma index         # incremental: embeds only new/changed items
zotgemma index --force # re-embed everything
```

Zotero may stay open: its database is copied to a temp directory and read there, `fulltext.sqlite` is opened
read-only, and nothing under the Zotero data dir is ever written. Items from every library (user and groups)
are indexed, including any attachment (PDF, HTML snapshot, EPUB, linked file) that has a
`.zotero-ft-cache` text file.

Cite keys come from Better BibTeX (`localhost:23119`), queried as `libraryID:itemKey`. When Zotero is not running
the keys already stored in the index are kept (`citekeys: cached (Zotero not running)`). For items BBT has no key
for, a `Citation Key:` line in the item's Extra field is used.

Device and batch size: `--device mps|cuda|cpu` (global, or on `index`/`search`/`eval`) or `ZOTGEMMA_DEVICE`;
default is MPS, then CUDA, then CPU. Embedding batch size defaults to 16 (mps), 64 (cuda), 8 (cpu);
override with `ZOTGEMMA_BATCH_SIZE`.

## Search

```sh
zotgemma search "passivity-based control of manipulators"
zotgemma search "lyapunov" --mode keyword     # dense | keyword | hybrid (default)
zotgemma search "consensus protocols" --year 2010: --type journalArticle -n 5
zotgemma search "sampling-based planning" --json
zotgemma show <citekey-or-item-key>
```

Keyword mode is sub-second. Dense and hybrid modes load the model on every CLI call (about 5 s);
the Phase 3 MCP server keeps it resident and amortizes that cost.

Each result carries cite key, year, authors, score, per-source ranks and a
`zotero://select/library/items/<KEY>` link. `--year` accepts `2018`, `2018:`, `:2005`,
`2010:2020`; `--collection` matches a substring of a collection path.

`--library` restricts results to `user`, a group ID, or a substring of a group name. Links to group items use
`zotero://select/groups/<groupID>/items/<KEY>`.

Environment variables: `ZOTGEMMA_ZOTERO_DIR`, `ZOTGEMMA_INDEX_DB`, `ZOTGEMMA_DEVICE`, `ZOTGEMMA_BATCH_SIZE`,
`ZOTGEMMA_BBT_URL`.

## Eval

`tests/golden.yaml` (in the repository) maps queries to expected Zotero item keys (any one counts); it is the
example for the author's own library. Bring your own:

```sh
zotgemma eval --golden path/to/golden.yaml   # default: tests/golden.yaml if it exists
```

Reports recall@5/@10 for keyword, dense and hybrid, and for Matryoshka truncation of
the dense vectors to 768/512/256 dimensions.

## Tests

```sh
uv run pytest
```

Tests build a synthetic Zotero 10 data directory (`tests/conftest.py`) and stub the embedder, so they need
neither Zotero nor the model.

## Layout

`zotgemma/`: `config`, `zotero_db`, `fulltext`, `citekeys`, `embedder`, `index`, `search`,
`discovery`, `evaluation`, `cli`; `mcp_server` is reserved for Phase 3 (`zotgemma-mcp`).
