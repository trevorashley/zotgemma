# zotgemma

A better Zotero search that leverages EmbeddingGemma 2: semantic + keyword search over a local Zotero library. Dense retrieval with
`google/embeddinggemma-2`, BM25 over item metadata and Zotero's own full-text index,
fused with reciprocal rank fusion (k=60). Read-only with respect to `~/Zotero`.
See `PLAN.md` for the design and phases.

## Install

Requires [uv](https://docs.astral.sh/uv/) and Python 3.14.

```sh
uv sync
uv run zotgemma --help
```

The first embedding run downloads the model (~1.5 GB) into the Hugging Face cache.

## Index

```sh
uv run zotgemma status        # library + index counts, Better BibTeX endpoint check
uv run zotgemma index         # incremental: embeds only new/changed items
uv run zotgemma index --force # re-embed everything
```

The index lives in `data/index.sqlite` (gitignored). Zotero may stay open: its database is
copied to a temp directory and read there; nothing under `~/Zotero` is ever written.
Cite keys come from Better BibTeX (`localhost:23119`), falling back to
`~/Zotero/better-bibtex.migrated` when Zotero is not running.

## Search

```sh
uv run zotgemma search "passivity-based control of manipulators"
uv run zotgemma search "lyapunov" --mode keyword     # dense | keyword | hybrid (default)
uv run zotgemma search "consensus protocols" --year 2010: --type journalArticle -n 5
uv run zotgemma search "sampling-based planning" --json
uv run zotgemma show <citekey-or-item-key>
```

Keyword mode is sub-second. Dense and hybrid modes load the model on every CLI call (about 5 s);
the Phase 3 MCP server keeps it resident and amortizes that cost.

Each result carries cite key, year, authors, score, per-source ranks and a
`zotero://select/library/items/<KEY>` link. `--year` accepts `2018`, `2018:`, `:2005`,
`2010:2020`; `--collection` matches a substring of a collection path.

Environment overrides: `ZOTGEMMA_ZOTERO_DIR`, `ZOTGEMMA_BBT_URL`, `ZOTGEMMA_INDEX_DB`.

## Eval

`tests/golden.yaml` maps queries to expected Zotero item keys (any one counts).

```sh
uv run zotgemma eval
```

Reports recall@5/@10 for keyword, dense and hybrid, and for Matryoshka truncation of
the dense vectors to 768/512/256 dimensions.

## Tests

```sh
uv run pytest
```

## Layout

`zotgemma/`: `config`, `zotero_db`, `fulltext`, `citekeys`, `embedder`, `index`, `search`,
`evaluation`, `cli`; `mcp_server` is reserved for Phase 3 (`zotgemma-mcp`).
