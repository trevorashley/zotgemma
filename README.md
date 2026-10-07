# zotero-search

Semantic + keyword search over a local Zotero library. Dense retrieval with
`google/embeddinggemma-2`, BM25 over item metadata and Zotero's own full-text index,
fused with reciprocal rank fusion (k=60). Read-only with respect to `~/Zotero`.
See `PLAN.md` for the design and phases.

## Install

Requires [uv](https://docs.astral.sh/uv/) and Python 3.14.

```sh
uv sync
uv run zsearch --help
```

The first embedding run downloads the model (~1.5 GB) into the Hugging Face cache.

## Index

```sh
uv run zsearch status        # library + index counts, Better BibTeX endpoint check
uv run zsearch index         # incremental: embeds only new/changed items
uv run zsearch index --force # re-embed everything
```

The index lives in `data/index.sqlite` (gitignored). Zotero may stay open: its database is
copied to a temp directory and read there; nothing under `~/Zotero` is ever written.
Cite keys come from Better BibTeX (`localhost:23119`), falling back to
`~/Zotero/better-bibtex.migrated` when Zotero is not running.

## Search

```sh
uv run zsearch search "passivity-based control of manipulators"
uv run zsearch search "lyapunov" --mode keyword     # dense | keyword | hybrid (default)
uv run zsearch search "consensus protocols" --year 2010: --type journalArticle -n 5
uv run zsearch search "sampling-based planning" --json
uv run zsearch show <citekey-or-item-key>
```

Each result carries cite key, year, authors, score, per-source ranks and a
`zotero://select/library/items/<KEY>` link. `--year` accepts `2018`, `2018:`, `:2005`,
`2010:2020`; `--collection` matches a substring of a collection path.

## Eval

`tests/golden.yaml` maps queries to expected Zotero item keys (any one counts).

```sh
uv run zsearch eval
```

Reports recall@5/@10 for keyword, dense and hybrid, and for Matryoshka truncation of
the dense vectors to 768/512/256 dimensions.

## Tests

```sh
uv run pytest
```

## Layout

`zsearch/`: `config`, `zotero_db`, `fulltext`, `citekeys`, `embedder`, `index`, `search`,
`evaluation`, `cli`; `mcp_server` is reserved for Phase 3 (`zsearch-mcp`).
