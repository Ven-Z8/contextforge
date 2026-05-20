from __future__ import annotations

import sys
from pathlib import Path

import typer

from contextforge.core.logger import get_logger

log = get_logger(__name__)
app = typer.Typer(
    name="contextforge",
    help="Evidence-preserving context optimizer for LLM systems.",
)


@app.command()
def optimize(
    query: str = typer.Option(..., "--query", "-q", help="The query to optimize context for"),
    sources: Path = typer.Option(
        ..., "--sources", "-s", help="Directory of .txt/.md files or single file"
    ),
    budget: int = typer.Option(8000, "--budget", "-b", help="Token budget (approximate)"),
    top_k: int = typer.Option(20, "--top-k", help="Candidates to consider"),
    top_n: int = typer.Option(5, "--top-n", help="Sources to include after reranking"),
) -> None:
    """Optimize context for a query from a directory of documents."""
    from contextforge import ContextEngine, Source

    if sources.is_dir():
        files = list(sources.glob("*.txt")) + list(sources.glob("*.md"))
    else:
        files = [sources]

    if not files:
        typer.echo(f"No .txt or .md files found in {sources}", err=True)
        raise typer.Exit(1) from None

    doc_sources = [Source(content=f.read_text(), path=f.name) for f in files]
    typer.echo(f"Loaded {len(doc_sources)} sources. Building context...\n")

    engine = ContextEngine(token_budget=budget, top_k=top_k, top_n=top_n)
    window = engine.build(query=query, sources=doc_sources)

    typer.echo("=" * 60)
    typer.echo(window.render())
    typer.echo("=" * 60)

    summary = window.compression_summary()
    typer.echo("\nSummary:")
    typer.echo(f"  Sources used:      {summary['total_chunks']}")
    typer.echo(f"  Chunks compressed: {summary['chunks_compressed']}")
    typer.echo(f"  Token budget:      {budget} (approximate)")
    typer.echo(f"  Tokens used:       {window.token_count()}")
    typer.echo(f"  Utilization:       {window.utilization(budget):.1%}")


@app.command()
def benchmark(
    n: int = typer.Option(50, "--n", help="Number of HotpotQA questions"),
    output: Path = typer.Option(Path("docs/benchmarks.md"), "--output", "-o"),
) -> None:
    """Run HotpotQA benchmark with real RAGAS metrics."""
    typer.echo(f"Running benchmark with {n} questions...")
    typer.echo("Requires: pip install contextforge[benchmark]")
    try:
        import subprocess
        subprocess.run(  # noqa: S603
            [sys.executable, "scripts/benchmark.py", "--n", str(n), "--output", str(output)],
            check=True,
        )
    except FileNotFoundError:
        typer.echo("scripts/benchmark.py not found. Run from project root.", err=True)
        raise typer.Exit(1) from None


if __name__ == "__main__":
    app()
