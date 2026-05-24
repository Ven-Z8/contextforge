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
from dataclasses import dataclass, field
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
FLOAT_TOLERANCE = 1e-9


@dataclass(frozen=True)
class StrategyObservation:
    strategy_key: str
    strategy: str
    evidence_hit: bool
    tokens: int
    utilization: float
    source_count: int
    recall_at_5_hit: bool
    recall_at_10_hit: bool
    mrr: float
    ndcg_at_10: float


@dataclass(frozen=True)
class ExampleObservation:
    example_id: str
    question: str
    answers: list[str]
    total_relevant: int
    strategies: dict[str, StrategyObservation] = field(default_factory=dict)


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def markdown_cell(text: str, limit: int = 100) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) > limit:
        collapsed = f"{collapsed[: limit - 3]}..."
    return collapsed.replace("|", "\\|")


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
) -> StrategyObservation:
    tokens = counter.count(rendered_context)
    evidence_hit = contains_answer(rendered_context, answers)
    positions = evidence_positions(ranked_contexts, answers)
    recall_at_5_hit = any(position < 5 for position in positions)
    recall_at_10_hit = any(position < 10 for position in positions)
    mrr = reciprocal_rank(positions)
    ndcg_10 = ndcg_at_k(positions, total_relevant, 10)

    result.examples += 1
    result.total_tokens += tokens
    result.total_sources += source_count
    result.utilizations.append(tokens / token_budget if token_budget else 0.0)
    result.latencies.append(latency_s)
    if evidence_hit:
        result.evidence_hits += 1

    if recall_at_5_hit:
        result.recall_at_5_hits += 1
    if recall_at_10_hit:
        result.recall_at_10_hits += 1
    result.total_mrr += mrr
    result.total_ndcg_10 += ndcg_10

    return StrategyObservation(
        strategy_key="",
        strategy=result.strategy,
        evidence_hit=evidence_hit,
        tokens=tokens,
        utilization=tokens / token_budget if token_budget else 0.0,
        source_count=source_count,
        recall_at_5_hit=recall_at_5_hit,
        recall_at_10_hit=recall_at_10_hit,
        mrr=mrr,
        ndcg_at_10=ndcg_10,
    )


def record_strategy(
    observations: dict[str, StrategyObservation],
    key: str,
    result: StrategyResult,
    **kwargs: object,
) -> None:
    observation = record_result(result, **kwargs)  # type: ignore[arg-type]
    observations[key] = StrategyObservation(
        strategy_key=key,
        strategy=observation.strategy,
        evidence_hit=observation.evidence_hit,
        tokens=observation.tokens,
        utilization=observation.utilization,
        source_count=observation.source_count,
        recall_at_5_hit=observation.recall_at_5_hit,
        recall_at_10_hit=observation.recall_at_10_hit,
        mrr=observation.mrr,
        ndcg_at_10=observation.ndcg_at_10,
    )


def evaluate_example(
    example: BenchmarkExample,
    *,
    scorer: SemanticScorer,
    engine: ContextEngine,
    counter: TokenCounter,
    token_budget: int,
    retrieved_k: int,
    raw_top_k: int,
    include_qdrant: bool,
    results: dict[str, StrategyResult],
    observations: list[ExampleObservation],
) -> None:
    total_relevant = len(evidence_positions(example.contexts, example.answers))
    strategy_observations: dict[str, StrategyObservation] = {}

    t0 = time.perf_counter()
    raw_ranked = example.contexts
    raw_contexts = raw_ranked[:raw_top_k]
    raw_rendered = "\n\n---\n\n".join(raw_contexts)
    record_strategy(
        strategy_observations,
        "raw",
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
    record_strategy(
        strategy_observations,
        "vector",
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
    record_strategy(
        strategy_observations,
        "bm25",
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

    if include_qdrant:
        from benchmarks.qdrant_backend import QdrantContextIndex

        qdrant_index = QdrantContextIndex(example.contexts, scorer._model)

        t0 = time.perf_counter()
        qdrant_dense_contexts = qdrant_index.rank_dense(example.question)
        qdrant_dense_rendered = "\n\n---\n\n".join(qdrant_dense_contexts[:raw_top_k])
        record_strategy(
            strategy_observations,
            "qdrant_dense",
            results["qdrant_dense"],
            rendered_context=qdrant_dense_rendered,
            ranked_contexts=qdrant_dense_contexts,
            total_relevant=total_relevant,
            source_count=len(qdrant_dense_contexts[:raw_top_k]),
            token_budget=token_budget,
            answers=example.answers,
            counter=counter,
            latency_s=time.perf_counter() - t0,
        )

        t0 = time.perf_counter()
        qdrant_hybrid_contexts = qdrant_index.rank_hybrid(example.question)
        qdrant_hybrid_rendered = "\n\n---\n\n".join(qdrant_hybrid_contexts[:raw_top_k])
        record_strategy(
            strategy_observations,
            "qdrant_hybrid",
            results["qdrant_hybrid"],
            rendered_context=qdrant_hybrid_rendered,
            ranked_contexts=qdrant_hybrid_contexts,
            total_relevant=total_relevant,
            source_count=len(qdrant_hybrid_contexts[:raw_top_k]),
            token_budget=token_budget,
            answers=example.answers,
            counter=counter,
            latency_s=time.perf_counter() - t0,
        )

        t0 = time.perf_counter()
        sources = [
            Source(
                content=context,
                source_id=f"{example.id}:qdrant-hybrid:{idx}",
                path=example.metadata.get("title"),
            )
            for idx, context in enumerate(qdrant_hybrid_contexts[:retrieved_k])
        ]
        window = engine.build(query=example.question, sources=sources)
        contextforge_contexts = [chunk.compressed_content for chunk in window.chunks]
        record_strategy(
            strategy_observations,
            "qdrant_contextforge",
            results["qdrant_contextforge"],
            rendered_context=window.render(),
            ranked_contexts=contextforge_contexts,
            total_relevant=total_relevant,
            source_count=window.source_count(),
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
    record_strategy(
        strategy_observations,
        "contextforge",
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
    observations.append(
        ExampleObservation(
            example_id=example.id,
            question=example.question,
            answers=example.answers,
            total_relevant=total_relevant,
            strategies=strategy_observations,
        )
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
        f"**Qdrant backend:** {'enabled' if args.include_qdrant else 'disabled'}",
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
    if args.include_qdrant:
        lines.extend(
            [
                "",
                "### Reading This Result",
                "",
                (
                    "Qdrant dense should track the local vector baseline because both use the "
                    "same embedding model."
                ),
                "",
                (
                    "Qdrant hybrid measures whether dense + sparse fusion improves evidence "
                    "retrieval before ContextForge compression. If hybrid recall is higher than "
                    "the ContextForge row, the next engineering target is compression/reranking "
                    "tuning that preserves the hybrid retriever's recall while reducing tokens."
                ),
            ]
        )
        vector = results.get("vector")
        hybrid = results.get("qdrant_hybrid")
        contextforge = results.get("contextforge")
        hybrid_contextforge = results.get("qdrant_contextforge")
        if vector and hybrid and contextforge and hybrid_contextforge:
            hybrid_recall_lift = hybrid.evidence_recall - vector.evidence_recall
            hybrid_token_delta = hybrid.avg_tokens - vector.avg_tokens
            contextforge_token_reduction = (
                1 - contextforge.avg_tokens / vector.avg_tokens
                if vector.avg_tokens
                else 0.0
            )
            hybrid_contextforge_recall_gap = (
                hybrid.evidence_recall - hybrid_contextforge.evidence_recall
            )
            lines.extend(
                [
                    "",
                    "### Current Run Notes",
                    "",
                    (
                        f"- Qdrant hybrid recall delta vs vector: "
                        f"{hybrid_recall_lift:+.3f}; avg token delta: "
                        f"{hybrid_token_delta:+.0f}."
                    ),
                    (
                        f"- Vector top-k + ContextForge token reduction vs vector: "
                        f"{contextforge_token_reduction:.1%}; recall delta: "
                        f"{contextforge.evidence_recall - vector.evidence_recall:+.3f}."
                    ),
                    (
                        f"- Qdrant hybrid + ContextForge recall gap vs Qdrant hybrid: "
                        f"{hybrid_contextforge_recall_gap:+.3f}. Treat this as a limitation, "
                        "not a win, until compression preserves the hybrid retriever's lift."
                    ),
                ]
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    console.print(f"[green]Saved to {output}[/green]")


def write_failure_analysis(
    observations: list[ExampleObservation], output: Path, args: argparse.Namespace
) -> None:
    def row(example: ExampleObservation, left: str, right: str) -> str:
        left_obs = example.strategies[left]
        right_obs = example.strategies[right]
        answers = ", ".join(example.answers[:3])
        return (
            f"| `{markdown_cell(example.example_id, 40)}` "
            f"| {markdown_cell(example.question, 80)} "
            f"| {markdown_cell(answers, 60)} "
            f"| {left_obs.evidence_hit} / {right_obs.evidence_hit} "
            f"| {left_obs.tokens} / {right_obs.tokens} |"
        )

    qdrant_losses = [
        example
        for example in observations
        if example.strategies.get("qdrant_hybrid")
        and example.strategies.get("qdrant_contextforge")
        and example.strategies["qdrant_hybrid"].evidence_hit
        and not example.strategies["qdrant_contextforge"].evidence_hit
    ]
    vector_losses = [
        example
        for example in observations
        if example.strategies["vector"].evidence_hit
        and not example.strategies["contextforge"].evidence_hit
    ]
    budget_violations = [
        (example, observation)
        for example in observations
        for key, observation in example.strategies.items()
        if key in {"contextforge", "qdrant_contextforge"} and observation.tokens > args.token_budget
    ]
    budget_violations.sort(key=lambda item: item[1].tokens, reverse=True)

    lines = [
        "## Qdrant Failure Analysis",
        "",
        f"**Dataset:** Natural Questions dev split, {args.n} examples",
        f"**Token budget:** {args.token_budget}",
        "",
        (
            "This file lists benchmark failures directly. It is meant for engineering "
            "triage, not marketing."
        ),
        "",
        "### Qdrant Hybrid Evidence Lost After ContextForge",
        "",
        (
            "| Example | Question | Answers | Hybrid Hit / ContextForge Hit | "
            "Hybrid Tokens / ContextForge Tokens |"
        ),
        "|---------|----------|---------|-------------------------------|-----------------------------------|",
    ]
    if qdrant_losses:
        lines.extend(
            row(example, "qdrant_hybrid", "qdrant_contextforge")
            for example in qdrant_losses[:20]
        )
    else:
        lines.append("| none | none | none | none | none |")

    lines.extend(
        [
            "",
            "### Vector Evidence Lost After ContextForge",
            "",
            (
                "| Example | Question | Answers | Vector Hit / ContextForge Hit | "
                "Vector Tokens / ContextForge Tokens |"
            ),
            "|---------|----------|---------|-------------------------------|-----------------------------------|",
        ]
    )
    if vector_losses:
        lines.extend(row(example, "vector", "contextforge") for example in vector_losses[:20])
    else:
        lines.append("| none | none | none | none | none |")

    lines.extend(
        [
            "",
            "### ContextForge Token Budget Violations",
            "",
            "| Example | Strategy | Question | Tokens | Utilization |",
            "|---------|----------|----------|--------|-------------|",
        ]
    )
    if budget_violations:
        for example, observation in budget_violations[:20]:
            lines.append(
                f"| `{markdown_cell(example.example_id, 40)}` "
                f"| {observation.strategy} "
                f"| {markdown_cell(example.question, 80)} "
                f"| {observation.tokens} "
                f"| {observation.utilization:.1%} |"
            )
    else:
        lines.append("| none | none | none | none | none |")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    console.print(f"[green]Saved failure analysis to {output}[/green]")


def write_gate_report(
    results: dict[str, StrategyResult],
    observations: list[ExampleObservation],
    output: Path,
    args: argparse.Namespace,
) -> None:
    def recall_drop(source: str, compressed: str) -> float:
        return results[source].evidence_recall - results[compressed].evidence_recall

    def token_reduction(source: str, compressed: str) -> float:
        source_tokens = results[source].avg_tokens
        if not source_tokens:
            return 0.0
        return 1 - results[compressed].avg_tokens / source_tokens

    contextforge_budget_violations = sum(
        1
        for example in observations
        for key, observation in example.strategies.items()
        if key in {"contextforge", "qdrant_contextforge"} and observation.tokens > args.token_budget
    )
    gates = [
        (
            "Vector recall preserved",
            recall_drop("vector", "contextforge") <= args.max_recall_drop + FLOAT_TOLERANCE,
            f"drop={recall_drop('vector', 'contextforge'):.3f}; max={args.max_recall_drop:.3f}",
        ),
        (
            "Vector token reduction",
            token_reduction("vector", "contextforge") >= args.min_token_reduction,
            (
                f"reduction={token_reduction('vector', 'contextforge'):.1%}; "
                f"min={args.min_token_reduction:.1%}"
            ),
        ),
        (
            "ContextForge token budget",
            contextforge_budget_violations == 0,
            f"violations={contextforge_budget_violations}; budget={args.token_budget}",
        ),
    ]
    if args.include_qdrant:
        gates.extend(
            [
                (
                    "Qdrant hybrid recall preserved",
                    recall_drop("qdrant_hybrid", "qdrant_contextforge")
                    <= args.max_recall_drop + FLOAT_TOLERANCE,
                    (
                        f"drop={recall_drop('qdrant_hybrid', 'qdrant_contextforge'):.3f}; "
                        f"max={args.max_recall_drop:.3f}"
                    ),
                ),
                (
                    "Qdrant hybrid token reduction",
                    token_reduction("qdrant_hybrid", "qdrant_contextforge")
                    >= args.min_token_reduction,
                    (
                        f"reduction={token_reduction('qdrant_hybrid', 'qdrant_contextforge'):.1%}; "
                        f"min={args.min_token_reduction:.1%}"
                    ),
                ),
            ]
        )

    lines = [
        "## Benchmark Gates",
        "",
        f"**Dataset:** Natural Questions dev split, {args.n} examples",
        "",
        "| Gate | Status | Evidence |",
        "|------|--------|----------|",
    ]
    for name, passed, evidence in gates:
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} | {evidence} |")
    lines.extend(
        [
            "",
            (
                "A failed gate blocks broad benchmark claims. It does not block publishing "
                "the result as an honest limitation."
            ),
        ]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    console.print(f"[green]Saved gate report to {output}[/green]")


def run(args: argparse.Namespace) -> tuple[dict[str, StrategyResult], list[ExampleObservation]]:
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
    }
    if args.include_qdrant:
        results.update(
            {
                "qdrant_dense": StrategyResult("Qdrant dense top-k"),
                "qdrant_hybrid": StrategyResult("Qdrant hybrid top-k"),
            }
        )
    results["contextforge"] = StrategyResult("Vector top-k + ContextForge")
    if args.include_qdrant:
        results["qdrant_contextforge"] = StrategyResult("Qdrant hybrid + ContextForge")
    observations: list[ExampleObservation] = []

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
            include_qdrant=args.include_qdrant,
            results=results,
            observations=observations,
        )
    return results, observations


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
    parser.add_argument("--include-qdrant", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("docs/public-benchmarks.md"))
    parser.add_argument(
        "--failure-output", type=Path, default=Path("docs/qdrant-failure-analysis.md")
    )
    parser.add_argument("--gates-output", type=Path, default=Path("docs/qdrant-gates.md"))
    parser.add_argument("--max-recall-drop", type=float, default=0.01)
    parser.add_argument("--min-token-reduction", type=float, default=0.30)
    args = parser.parse_args()

    results, observations = run(args)
    print_results(results)
    write_markdown(results, args.output, args)
    if args.include_qdrant:
        write_failure_analysis(observations, args.failure_output, args)
        write_gate_report(results, observations, args.gates_output, args)


if __name__ == "__main__":
    main()
