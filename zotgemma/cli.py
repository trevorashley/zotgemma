"""``zotgemma`` command-line interface."""

from __future__ import annotations

import json
import logging
import importlib.metadata
import sys
import time
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from .models import zotero_link
from . import citekeys, config, discovery, fulltext, index, search as search_mod, zotero_db

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
    except (FileNotFoundError, RuntimeError) as e:
        raise _fail(str(e))


_config_error: str | None = None
DeviceOpt = Annotated[Optional[str], typer.Option("--device", help="mps | cuda | cpu (default: auto; env ZOTGEMMA_DEVICE).")]


def _version_cb(value: bool) -> None:
    if value:
        try:
            v = importlib.metadata.version("zotgemma")
        except importlib.metadata.PackageNotFoundError:
            v = "unknown"
        console.print(f"zotgemma {v}")
        raise typer.Exit()


def _need_config() -> None:
    if _config_error:
        raise _fail(_config_error)


def _set_device(device: str | None) -> None:
    if device is not None:
        try:
            config.set_device(device)
        except ValueError as e:
            raise _fail(str(e))


@app.callback()
def _main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Also log info messages (warnings are always shown).")] = False,
    zotero_dir: Annotated[Optional[str], typer.Option("--zotero-dir", help="Zotero data directory (default: auto-discovered).")] = None,
    index_db: Annotated[Optional[str], typer.Option("--index", help="Index file (default: per-library, under the user data dir).")] = None,
    device: DeviceOpt = None,
    version: Annotated[bool, typer.Option("--version", callback=_version_cb, is_eager=True, help="Show the version and exit.")] = False,
) -> None:
    global _config_error
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(levelname)s %(message)s")
    try:
        config.configure(zotero_dir=zotero_dir, index_db=index_db, device=device)
        _config_error = None
    except (discovery.DiscoveryError, ValueError) as e:
        _config_error = str(e)  # reported by the command that needs it, so --help still works


@app.command()
def status(missing: Annotated[bool, typer.Option("--missing", help="List attachments without extracted text.")] = False) -> None:
    """Show library and index counts."""
    _need_config()
    try:
        with zotero_db.snapshot() as z:
            counts = zotero_db.library_counts(z)
            items = zotero_db.load_items(z)
            libs = zotero_db.load_libraries(z)
            userdata = zotero_db.userdata_version(z)
    except zotero_db.ZoteroDBError as e:
        raise _fail(str(e))
    atts = [a for i in items for a in i.attachments]
    by_type = zotero_db.attachment_type_counts(atts)
    caches = sum(1 for p in config.STORAGE_DIR.glob("*/.zotero-ft-cache")) if config.STORAGE_DIR.exists() else 0
    try:
        indexed = fulltext.indexed_attachment_ids()
    except Exception:  # noqa: BLE001 - status must not die on a full-text problem
        indexed = set()
    no_text = [a for a in atts if not fulltext.cache_path(a.key).exists()]

    console.print(f"Zotero data dir: {config.ZOTERO_DIR}  [dim](from {escape(config.DATA_DIR_SOURCE)}; schema userdata={userdata})[/dim]", soft_wrap=True)
    console.print(f"Index file:      {config.INDEX_DB}  [dim]({'default location' if config.INDEX_DB_IS_DEFAULT else 'override'})[/dim]", soft_wrap=True)
    t = Table(title="Zotero library", show_header=False)
    rows: list[tuple[str, object]] = [
        ("libraries", ", ".join(f"{l.name or l.type} (id {l.library_id}, {l.type})" for l in libs.values())),
        ("bibliographic items", counts["bibliographic_items"]),
        ("standalone attachments (indexed as items)", sum(1 for i in items if i.standalone)),
        ("attachments (all types)", counts["attachments"]),
        ("attachments with a text cache", sum(by_type.values())),
        *[(f"  {ct}", n) for ct, n in by_type.items()],
        (".zotero-ft-cache files", caches),
        ("attachments in Zotero FTS index", len(indexed)),
        ("PDFs with no text cache", len(no_text)),
        ("highlight annotations", counts["annotations"]),
        ("collections", counts["collections"]),
        ("tags", counts["tags"]),
    ]
    for k, v in rows:
        t.add_row(k, f"{v:,}" if isinstance(v, int) else str(v))
    console.print(t)
    console.print("Better BibTeX endpoint: " + ("[green]up[/green]" if citekeys.bbt_available()
                                                else "[yellow]down (cached cite keys will be kept)[/yellow]"))

    if config.INDEX_DB.exists() or (config.INDEX_DB_IS_DEFAULT and config.LEGACY_INDEX_DB.exists()):
        conn = _open_index()  # also migrates a legacy data/index.sqlite
        c = index.index_counts(conn)
        t2 = Table(title="Index", show_header=False)
        for k, v in c.items():
            t2.add_row(k.replace("_", " "), f"{v:,}")
        for k in ("model_id", "dtype", "last_sync"):
            t2.add_row(k.replace("_", " "), index.get_meta(conn, k) or "-")
        console.print(t2)
        conn.close()
    else:
        console.print(f"[yellow]No index yet at {config.INDEX_DB}; run `zotgemma index`.[/yellow]")
    if missing:
        for a in no_text:
            console.print(f"  no text: {a.key}  {a.filename}")


@app.command("index")
def index_cmd(force: Annotated[bool, typer.Option("--force", help="Re-embed everything.")] = False,
              device: DeviceOpt = None) -> None:
    """Sync the index with Zotero (embeds only new or changed items)."""
    _need_config()
    _set_device(device)
    t0 = time.time()
    with Progress(TextColumn("embedding"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
                  TimeRemainingColumn(), console=console, transient=True) as prog:
        task = prog.add_task("e", total=None)

        def cb(done: int, total: int) -> None:
            prog.update(task, completed=done, total=total)

        try:
            s = index.sync(progress=cb, force=force)
        except (zotero_db.ZoteroDBError, RuntimeError) as e:
            raise _fail(str(e))
    console.print(f"items: {s.total:,}  embedded: {s.embedded:,}  removed: {s.removed:,}  "
                  f"citekeys: {s.citekeys_found:,} (source: {s.citekey_source}; {s.citekeys_from_extra:,} from extra)")
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
    library: Annotated[Optional[str], typer.Option("--library", help="user, a group ID, or a group-name substring.")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
    device: DeviceOpt = None,
) -> None:
    """Search the library."""
    _need_config()
    _set_device(device)
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
    if library:
        where, args = search_mod.library_clause(library, conn)
        if not conn.execute(f"SELECT 1 FROM items WHERE {where} LIMIT 1", args).fetchone():
            libs = "; ".join(f"{n or 'user'} (group {g})" if g else f"{n or 'user'} (user)"
                             for g, n, _ in search_mod.list_libraries(conn))
            conn.close()
            raise _fail(f"no items in library {library!r}; indexed libraries: {libs}")
    try:
        hits = search_mod.search(conn, query, limit=limit, mode=mode, year_from=yf, year_to=yt,
                                 item_type=item_type, collection=collection, library=library)
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
    _need_config()
    conn = _open_index()
    row = conn.execute("SELECT * FROM items WHERE citekey = ? OR key = ? ORDER BY library_id", (ref, ref)).fetchone()
    att_ids = [r[0] for r in conn.execute(
        "SELECT attachment_id FROM attachments WHERE item_id = ? ORDER BY is_best DESC, attachment_id", (row["item_id"],))] if row else []
    conn.close()
    if row is None:
        raise _fail(f"No item with citekey or key {ref!r} in the index.")
    console.print(f"[bold]{escape(row['title'])}[/bold]")
    for label, val in [("citekey", row["citekey"]), ("key", row["key"]), ("type", row["item_type"]),
                       ("authors", row["authors"]), ("year", row["year"]), ("venue", row["venue"]),
                       ("DOI", row["doi"]), ("URL", row["url"]),
                       ("library", row["library_name"]),
                       ("link", zotero_link(row["key"], row["group_id"])),
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
def eval_cmd(golden_path: Annotated[Optional[Path], typer.Option(
        "--golden", help="Golden-set YAML (default: tests/golden.yaml if it exists).")] = None,
        device: DeviceOpt = None) -> None:
    """Run a golden set and report recall@5/@10 per mode and truncate_dim."""
    from . import evaluation

    _need_config()
    _set_device(device)
    path = golden_path or (config.GOLDEN_YAML if config.GOLDEN_YAML.is_file() else None)
    if path is None:
        raise _fail("No golden set: pass --golden PATH (YAML of queries -> expected item keys; "
                    "see tests/golden.yaml in the repository for the format).")
    try:
        golden = evaluation.load_golden(path)
    except (OSError, KeyError, TypeError, ValueError) as e:
        raise _fail(f"Cannot load golden set {path}: {e}")
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
