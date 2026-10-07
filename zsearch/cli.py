"""``zsearch`` command-line interface."""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from . import citekeys, config, fulltext, index, search as search_mod, zotero_db

app = typer.Typer(help="Semantic + keyword search over your local Zotero library.", no_args_is_help=True,
                  add_completion=False)
console = Console()
err = Console(stderr=True)


def _fail(msg: str, code: int = 1) -> typer.Exit:
    err.print(f"[red]error:[/red] {escape(msg)}")
    return typer.Exit(code)


def _open_index():
    try:
        return index.open_index()
    except FileNotFoundError as e:
        raise _fail(str(e))


@app.callback()
def _main(verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Log warnings and info.")] = False) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.ERROR, format="%(levelname)s %(message)s")


@app.command()
def status(missing: Annotated[bool, typer.Option("--missing", help="List PDFs without extracted text.")] = False) -> None:
    """Show library and index counts."""
    try:
        with zotero_db.snapshot() as z:
            counts = zotero_db.library_counts(z)
            items = zotero_db.load_items(z)
    except zotero_db.ZoteroDBError as e:
        raise _fail(str(e))
    pdfs = [a for i in items for a in i.attachments]
    caches = sum(1 for p in config.STORAGE_DIR.glob("*/.zotero-ft-cache")) if config.STORAGE_DIR.exists() else 0
    try:
        indexed = fulltext.indexed_attachment_ids()
    except Exception:  # noqa: BLE001 - status must not die on a full-text problem
        indexed = set()
    no_text = [a for a in pdfs if not fulltext.cache_path(a.key).exists()]

    t = Table(title="Zotero library", show_header=False)
    for k, v in [
        ("bibliographic items", counts["bibliographic_items"]),
        ("standalone PDFs (indexed as items)", sum(1 for i in items if i.standalone)),
        ("PDF attachments", counts["pdf_attachments"]),
        (".zotero-ft-cache files", caches),
        ("attachments in Zotero FTS index", len(indexed)),
        ("PDFs with no text cache", len(no_text)),
        ("highlight annotations", counts["annotations"]),
        ("collections", counts["collections"]),
        ("tags", counts["tags"]),
    ]:
        t.add_row(k, f"{v:,}")
    console.print(t)
    console.print(f"Better BibTeX endpoint: {'[green]up[/green]' if citekeys.bbt_available() else '[yellow]down (fallback: better-bibtex.migrated)[/yellow]'}")

    if config.INDEX_DB.exists():
        conn = index.open_index()
        c = index.index_counts(conn)
        t2 = Table(title=f"Index ({config.INDEX_DB})", show_header=False)
        for k, v in c.items():
            t2.add_row(k.replace("_", " "), f"{v:,}")
        for k in ("model_id", "dtype", "last_sync"):
            t2.add_row(k.replace("_", " "), index.get_meta(conn, k) or "-")
        console.print(t2)
        conn.close()
    else:
        console.print(f"[yellow]No index yet at {config.INDEX_DB}; run `zsearch index`.[/yellow]")
    if missing:
        for a in no_text:
            console.print(f"  no text: {a.key}  {a.filename}")


@app.command("index")
def index_cmd(force: Annotated[bool, typer.Option("--force", help="Re-embed everything.")] = False) -> None:
    """Sync the index with Zotero (embeds only new or changed items)."""
    t0 = time.time()
    with Progress(TextColumn("embedding"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
                  TimeRemainingColumn(), console=console, transient=True) as prog:
        task = prog.add_task("e", total=None)

        def cb(done: int, total: int) -> None:
            prog.update(task, completed=done, total=total)

        try:
            s = index.sync(progress=cb, force=force)
        except zotero_db.ZoteroDBError as e:
            raise _fail(str(e))
    console.print(f"items: {s.total:,}  embedded: {s.embedded:,}  removed: {s.removed:,}  "
                  f"citekeys: {s.citekeys_found:,} (source: {s.citekey_source})")
    if s.embedded:
        console.print(f"model: {config.MODEL_ID} on {s.device} ({s.dtype})")
        console.print(f"embedding time: {s.embed_seconds:.1f}s  "
                      f"{s.embedded / s.embed_seconds:.2f} items/s  {s.tokens / s.embed_seconds:,.0f} tokens/s  "
                      f"({s.tokens:,} tokens)")
    console.print(f"total wall-clock: {time.time() - t0:.1f}s")


@app.command("search")
def search_cmd(
    query: Annotated[str, typer.Argument(help="Natural-language query.")],
    mode: Annotated[str, typer.Option("--mode", "-m", help="dense | keyword | hybrid")] = "hybrid",
    limit: Annotated[int, typer.Option("--limit", "-n", min=1, help="Max results (>= 1).")] = 10,
    year: Annotated[Optional[str], typer.Option("--year", help="e.g. 2018, 2018:, :2005, 2010:2020")] = None,
    item_type: Annotated[Optional[str], typer.Option("--type", help="e.g. journalArticle, book")] = None,
    collection: Annotated[Optional[str], typer.Option("--collection", help="Substring of a collection path.")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
) -> None:
    """Search the library."""
    if mode not in search_mod.MODES:
        raise _fail(f"--mode must be one of {', '.join(search_mod.MODES)}")
    try:
        yf, yt = search_mod.parse_year_range(year)
    except ValueError:
        raise _fail(f"bad --year {year!r}; use forms like 2018, 2018:, :2005, 2010:2020")
    if not query.strip():
        raise _fail("query is empty")
    if yf is not None and yt is not None and yf > yt:
        raise _fail(f"--year range is inverted ({yf} > {yt})")
    conn = _open_index()
    if item_type:
        valid = [r[0] for r in conn.execute("SELECT DISTINCT item_type FROM items ORDER BY 1")]
        if item_type not in valid:
            conn.close()
            raise _fail(f"unknown --type {item_type!r}; valid types: {', '.join(valid)}")
    try:
        hits = search_mod.search(conn, query, limit=limit, mode=mode, year_from=yf, year_to=yt,
                                 item_type=item_type, collection=collection)
    except RuntimeError as e:
        raise _fail(str(e))
    finally:
        conn.close()
    if as_json:
        json.dump([h.to_dict() for h in hits], sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
        return
    if not hits:
        console.print("[yellow]No results.[/yellow]")
        return
    t = Table(show_lines=True)
    for col in ("#", "score", "year", "authors", "title / citekey / link"):
        t.add_column(col, overflow="fold")
    for n, h in enumerate(hits, 1):
        ranks = " ".join(f"{k}:{v}" for k, v in h.ranks.items())
        t.add_row(str(n), f"{h.score:.4f}", str(h.year or ""), escape(search_mod.short_authors(h.authors)),
                  f"{escape(h.title)}\n[dim]{h.citekey or '(no citekey)'}  [{ranks}]\n{h.link}[/dim]")
    console.print(t)


@app.command()
def show(ref: Annotated[str, typer.Argument(help="Better BibTeX citekey or Zotero item key.")]) -> None:
    """Show metadata, abstract, collections, tags, attachment and annotations for one item."""
    conn = _open_index()
    row = conn.execute("SELECT * FROM items WHERE citekey = ? OR key = ?", (ref, ref)).fetchone()
    att_ids = [r[0] for r in conn.execute(
        "SELECT attachment_id FROM attachments WHERE item_id = ? ORDER BY is_best DESC, attachment_id", (row["item_id"],))] if row else []
    conn.close()
    if row is None:
        raise _fail(f"No item with citekey or key {ref!r} in the index.")
    console.print(f"[bold]{escape(row['title'])}[/bold]")
    for label, val in [("citekey", row["citekey"]), ("key", row["key"]), ("type", row["item_type"]),
                       ("authors", row["authors"]), ("year", row["year"]), ("venue", row["venue"]),
                       ("DOI", row["doi"]), ("URL", row["url"]),
                       ("link", f"zotero://select/library/items/{row['key']}"),
                       ("attachment", row["attachment_path"])]:
        if val:
            console.print(f"[cyan]{label}:[/cyan] {escape(str(val))}")
    if row["collections"]:
        console.print("[cyan]collections:[/cyan] " + escape("; ".join(row["collections"].split(index.TAG_SEP))))
    if row["tags"]:
        console.print("[cyan]tags:[/cyan] " + escape(", ".join(row["tags"].split(index.TAG_SEP))))
    if row["abstract"]:
        console.print(f"\n[cyan]abstract:[/cyan]\n{escape(row['abstract'])}")
    if att_ids:
        try:
            with zotero_db.snapshot() as z:
                anns = zotero_db.load_annotations(z, att_ids)
        except zotero_db.ZoteroDBError as e:
            err.print(f"[yellow]annotations unavailable: {escape(str(e))}[/yellow]")
            anns = []
        if anns:
            console.print(f"\n[cyan]annotations ({len(anns)}):[/cyan]")
            for a in anns:
                pg = f"p.{a.page_label} " if a.page_label else ""
                console.print(f"  - {pg}{escape(a.text)}" + (f"  [dim]// {escape(a.comment)}[/dim]" if a.comment else ""))


@app.command("eval")
def eval_cmd() -> None:
    """Run tests/golden.yaml and report recall@5/@10 per mode and truncate_dim."""
    from . import evaluation

    try:
        golden = evaluation.load_golden()
    except (OSError, KeyError) as e:
        raise _fail(f"Cannot load {config.GOLDEN_YAML}: {e}")
    conn = _open_index()
    try:
        res = evaluation.run_eval(conn, golden)
    except RuntimeError as e:
        raise _fail(str(e))
    finally:
        conn.close()
    kinds = sorted({g.kind for g in golden})
    t = Table(title=f"Golden set: {len(golden)} queries (recall = mean fraction of expected items; hit = any expected)")
    for c in ("config", "recall@5", "recall@10", "hit@5", "hit@10", *[f"hit@10 {k}" for k in kinds]):
        t.add_column(c, justify="right" if c != "config" else "left")
    for label, r in res.items():
        t.add_row(label, f"{r['recall@5']:.3f}", f"{r['recall@10']:.3f}", f"{r['hit@5']:.3f}", f"{r['hit@10']:.3f}",
                  *[f"{r['by_kind'].get(k, float('nan')):.2f}" for k in kinds])
    console.print(t)
    for label in ("keyword", "dense@768", "hybrid@768"):
        if res[label]["misses"]:
            console.print(f"[dim]{label} misses@10:[/dim] " + escape(" | ".join(res[label]["misses"])))


if __name__ == "__main__":
    app()
