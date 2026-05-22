#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.corpora.natural_questions import iter_natural_questions
from benchmarks.schemas import BenchmarkExample, StrategyResult
from contextforge import ContextEngine, Source
from contextforge.budget import TokenCounter
from contextforge.scorer import SemanticScorer

console = Console()


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def contains_answer(text: str, answers: list[str]) -> bool:
    haystack = normalize(text)
    return any(normalize(answer) in haystack for answer in answers)


def rank_contexts(query: str, contexts: list[str], scorer: SemanticScorer) -> list[str]:
    scores = scorer.score(query, contexts)
    ranked = sorted(zip(contexts, scores, strict=True), key=lambda item: item[1], reverse=True)
    return [context for context, _score in ranked]


def record_result(
    result: StrategyResult,
    *,
    rendered_context: str,
    source_count: int,
    token_budget: int,
    answers: list[str],
    counter: TokenCounter,
    latency_s: float,
) -> None:
    tokens = counter.count(rendered_context)
    result.examples += 1
    result.total_tokens += tokens
    result.total_sources += source_count
    result.utilizations.append(tokens / token_budget if token_budget else 0.0)
    result.latencies.append(latency_s)
    if contains_answer(rendered_context, answers):
        result.evidence_hits += 1


def evaluate_example(
    example: BenchmarkExample,
    *,
    scorer: SemanticScorer,
    engine: ContextEngine,
    counter: TokenCounter,
    token_budget: int,
    retrieved_k: int,
    raw_top_k: int,
    results: dict[str, StrategyResult],
) -> None:
    t0 = time.perf_counter()
    raw_contexts = example.contexts[:raw_top_k]
    raw_rendered = "\n\n---\n\n".join(raw_contexts)
    record_result(
        results["raw"],
        rendered_context=raw_rendered,
        source_count=len(raw_contexts),
        token_budget=token_budget,
        answers=example.answers,
        counter=counter,
        latency_s=time.perf_counter() - t0,
    )

    t0 = time.perf_counter()
    ranked_contexts = rank_contexts(example.question, example.contexts, scorer)
    vector_contexts = ranked_contexts[:raw_top_k]
    vector_rendered = "\n\n---\n\n".join(vector_contexts)
    record_result(
        results["vector"],
        rendered_context=vector_rendered,
        source_count=len(vector_contexts),
        token_budget=token_budget,
        answers=example.answers,
        counter=counter,
        latency_s=time.perf_counter() - t0,
    )

    t0 = time.perf_counter()
    retrieved_contexts = ranked_contexts[:retrieved_k]
    sources = [
        Source(content=context, source_id=f"{example.id}:{idx}", path=example.metadata.get("title"))
        for idx, context in enumerate(retrieved_contexts)
    ]
    window = engine.build(query=example.question, sources=sources)
    record_result(
        results["contextforge"],
        rendered_context=window.render(),
        source_count=window.source_count(),
        token_budget=token_budget,
        answers=example.answers,
        counter=counter,
        latency_s=time.perf_counter() - t0,
    )


def print_results(results: dict[str, StrategyResult]) -> None:
    table = Table(title="Natural Questions Public Retrieval Benchmark")
    table.add_column("Strategy")
    table.add_column("Evidence Recall", justify="right")
    table.add_column("Avg Tokens", justify="right")
    table.add_column("Utilization", justify="right")
    table.add_column("Avg Sources", justify="right")
    table.add_column("Latency p95", justify="right")
    for result in results.values():
        table.add_row(
            result.strategy,
            f"{result.evidence_recall:.3f}",
            f"{result.avg_tokens:.0f}",
            f"{result.avg_utilization:.1%}",
            f"{result.avg_sources:.1f}",
            f"{result.latency_p95:.2f}s",
        )
    console.print(table)


def write_markdown(results: dict[str, StrategyResult], output: Path, n: int) -> None:
    lines = [
        "## Public Retrieval Benchmark",
        "",
        f"**Dataset:** Natural Questions dev split, {n} examples",
        "**Metric:** Evidence recall = selected context contains a gold short answer string",
        (
            "**Note:** This benchmark evaluates public retrieval context selection, "
            "not final LLM answers."
        ),
        "",
        "| Strategy | Evidence Recall | Avg Tokens | Utilization | Avg Sources | Latency p95 |",
        "|----------|-----------------|------------|-------------|-------------|-------------|",
    ]
    for result in results.values():
        lines.append(
            f"| {result.strategy} | {result.evidence_recall:.3f} | {result.avg_tokens:.0f} "
            f"| {result.avg_utilization:.1%} | {result.avg_sources:.1f} "
            f"| {result.latency_p95:.2f}s |"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    console.print(f"[green]Saved to {output}[/green]")


def run(args: argparse.Namespace) -> dict[str, StrategyResult]:
    if args.dataset != "natural_questions":
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    scorer = SemanticScorer()
    engine = ContextEngine(
        token_budget=args.token_budget,
        top_k=args.retrieved_k,
        top_n=args.contextforge_top_n,
        scorer=scorer,
    )
    counter = TokenCounter()
    results = {
        "raw": StrategyResult("Raw candidate top-k"),
        "vector": StrategyResult("Vector top-k"),
        "contextforge": StrategyResult("Vector top-k + ContextForge"),
    }

    examples = iter_natural_questions(
        n=args.n,
        split=args.split,
        config=args.config,
        max_contexts=args.max_contexts,
    )
    for idx, example in enumerate(examples, start=1):
        console.print(f"[{idx}/{args.n}] {example.question[:80]}")
        evaluate_example(
            example,
            scorer=scorer,
            engine=engine,
            counter=counter,
            token_budget=args.token_budget,
            retrieved_k=args.retrieved_k,
            raw_top_k=args.raw_top_k,
            results=results,
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="ContextForge public retrieval benchmark")
    parser.add_argument("--dataset", default="natural_questions")
    parser.add_argument("--n", type=int, default=25)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--config", default="dev")
    parser.add_argument("--max-contexts", type=int, default=60)
    parser.add_argument("--raw-top-k", type=int, default=5)
    parser.add_argument("--retrieved-k", type=int, default=20)
    parser.add_argument("--contextforge-top-n", type=int, default=5)
    parser.add_argument("--token-budget", type=int, default=4000)
    parser.add_argument("--output", type=Path, default=Path("docs/public-benchmarks.md"))
    args = parser.parse_args()

    results = run(args)
    print_results(results)
    write_markdown(results, args.output, args.n)


if __name__ == "__main__":
    main()
