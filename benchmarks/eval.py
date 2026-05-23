#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import math
import random
import re
import sys
import time
from collections import Counter
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

TOKEN_RE = re.compile(r"\b\w+\b")


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def contains_answer(text: str, answers: list[str]) -> bool:
    haystack = normalize(text)
    return any(normalize(answer) in haystack for answer in answers)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def shuffle_contexts(example: BenchmarkExample, seed: int) -> BenchmarkExample:
    contexts = list(example.contexts)
    rng = random.Random(f"{seed}:{example.id}")
    rng.shuffle(contexts)
    metadata = dict(example.metadata)
    metadata["candidate_order"] = "shuffled"
    metadata["seed"] = str(seed)
    return BenchmarkExample(
        id=example.id,
        question=example.question,
        answers=example.answers,
        contexts=contexts,
        metadata=metadata,
    )


def rank_contexts(query: str, contexts: list[str], scorer: SemanticScorer) -> list[str]:
    scores = scorer.score(query, contexts)
    ranked = sorted(zip(contexts, scores, strict=True), key=lambda item: item[1], reverse=True)
    return [context for context, _score in ranked]


def rank_contexts_bm25(query: str, contexts: list[str]) -> list[str]:
    query_terms = tokenize(query)
    tokenized_contexts = [tokenize(context) for context in contexts]
    doc_freq: Counter[str] = Counter()
    for tokens in tokenized_contexts:
        doc_freq.update(set(tokens))

    doc_count = len(tokenized_contexts)
    avg_len = (
        sum(len(tokens) for tokens in tokenized_contexts) / doc_count if tokenized_contexts else 0.0
    )
    k1 = 1.5
    b = 0.75

    scored: list[tuple[str, float]] = []
    for context, tokens in zip(contexts, tokenized_contexts, strict=True):
        term_freq = Counter(tokens)
        score = 0.0
        doc_len = len(tokens)
        for term in query_terms:
            if not term_freq[term]:
                continue
            idf = math.log(1 + (doc_count - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            denominator = term_freq[term] + k1 * (1 - b + b * doc_len / avg_len) if avg_len else 1.0
            score += idf * (term_freq[term] * (k1 + 1)) / denominator
        scored.append((context, score))

    ranked = sorted(scored, key=lambda item: item[1], reverse=True)
    return [context for context, _score in ranked]


def evidence_positions(contexts: list[str], answers: list[str]) -> list[int]:
    return [idx for idx, context in enumerate(contexts) if contains_answer(context, answers)]


def reciprocal_rank(positions: list[int]) -> float:
    return 1 / (positions[0] + 1) if positions else 0.0


def ndcg_at_k(positions: list[int], total_relevant: int, k: int) -> float:
    if total_relevant <= 0:
        return 0.0
    dcg = sum(1 / math.log2(position + 2) for position in positions if position < k)
    ideal_relevant = min(total_relevant, k)
    ideal_dcg = sum(1 / math.log2(rank + 2) for rank in range(ideal_relevant))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def record_result(
    result: StrategyResult,
    *,
    rendered_context: str,
    ranked_contexts: list[str],
    total_relevant: int,
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

    positions = evidence_positions(ranked_contexts, answers)
    if any(position < 5 for position in positions):
        result.recall_at_5_hits += 1
    if any(position < 10 for position in positions):
        result.recall_at_10_hits += 1
    result.total_mrr += reciprocal_rank(positions)
    result.total_ndcg_10 += ndcg_at_k(positions, total_relevant, 10)


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
    total_relevant = len(evidence_positions(example.contexts, example.answers))

    t0 = time.perf_counter()
    raw_ranked = example.contexts
    raw_contexts = raw_ranked[:raw_top_k]
    raw_rendered = "\n\n---\n\n".join(raw_contexts)
    record_result(
        results["raw"],
        rendered_context=raw_rendered,
        ranked_contexts=raw_ranked,
        total_relevant=total_relevant,
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
        ranked_contexts=ranked_contexts,
        total_relevant=total_relevant,
        source_count=len(vector_contexts),
        token_budget=token_budget,
        answers=example.answers,
        counter=counter,
        latency_s=time.perf_counter() - t0,
    )

    t0 = time.perf_counter()
    bm25_ranked_contexts = rank_contexts_bm25(example.question, example.contexts)
    bm25_contexts = bm25_ranked_contexts[:raw_top_k]
    bm25_rendered = "\n\n---\n\n".join(bm25_contexts)
    record_result(
        results["bm25"],
        rendered_context=bm25_rendered,
        ranked_contexts=bm25_ranked_contexts,
        total_relevant=total_relevant,
        source_count=len(bm25_contexts),
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
    contextforge_contexts = [chunk.compressed_content for chunk in window.chunks]
    record_result(
        results["contextforge"],
        rendered_context=window.render(),
        ranked_contexts=contextforge_contexts,
        total_relevant=total_relevant,
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
    table.add_column("R@5", justify="right")
    table.add_column("R@10", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column("NDCG@10", justify="right")
    table.add_column("Avg Tokens", justify="right")
    table.add_column("Tokens/Hit", justify="right")
    table.add_column("Utilization", justify="right")
    table.add_column("Avg Sources", justify="right")
    table.add_column("Latency p50", justify="right")
    table.add_column("Latency p95", justify="right")
    for result in results.values():
        table.add_row(
            result.strategy,
            f"{result.evidence_recall:.3f}",
            f"{result.recall_at_5:.3f}",
            f"{result.recall_at_10:.3f}",
            f"{result.mrr:.3f}",
            f"{result.ndcg_at_10:.3f}",
            f"{result.avg_tokens:.0f}",
            f"{result.tokens_per_evidence_hit:.0f}",
            f"{result.avg_utilization:.1%}",
            f"{result.avg_sources:.1f}",
            f"{result.latency_p50:.2f}s",
            f"{result.latency_p95:.2f}s",
        )
    console.print(table)


def write_markdown(
    results: dict[str, StrategyResult], output: Path, args: argparse.Namespace
) -> None:
    lines = [
        "## Public Retrieval Benchmark",
        "",
        f"**Dataset:** Natural Questions dev split, {args.n} examples",
        f"**Candidate order:** shuffled with fixed seed `{args.seed}`",
        "**Task:** context selection over public Natural Questions long-answer candidates",
        (
            "**Evidence hit:** selected context contains a gold short answer string. "
            "This benchmark does not judge final LLM answer quality."
        ),
        "",
        (
            "| Strategy | Evidence Recall | R@5 | R@10 | MRR | NDCG@10 | Avg Tokens | "
            "Tokens/Hit | Utilization | Avg Sources | Latency p50 | Latency p95 |"
        ),
        (
            "|----------|-----------------|-----|------|-----|---------|------------|"
            "------------|-------------|-------------|-------------|-------------|"
        ),
    ]
    for result in results.values():
        lines.append(
            f"| {result.strategy} | {result.evidence_recall:.3f} "
            f"| {result.recall_at_5:.3f} | {result.recall_at_10:.3f} "
            f"| {result.mrr:.3f} | {result.ndcg_at_10:.3f} "
            f"| {result.avg_tokens:.0f} | {result.tokens_per_evidence_hit:.0f} "
            f"| {result.avg_utilization:.1%} | {result.avg_sources:.1f} "
            f"| {result.latency_p50:.2f}s | {result.latency_p95:.2f}s |"
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
        "raw": StrategyResult("Shuffled candidate top-k"),
        "bm25": StrategyResult("BM25 top-k"),
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
        example = shuffle_contexts(example, args.seed)
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
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", type=Path, default=Path("docs/public-benchmarks.md"))
    args = parser.parse_args()

    results = run(args)
    print_results(results)
    write_markdown(results, args.output, args)


if __name__ == "__main__":
    main()
