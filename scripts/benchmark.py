#!/usr/bin/env python3
"""ContextForge HotpotQA Benchmark — real RAGAS metrics.

Usage: uv run python scripts/benchmark.py [--n 100] [--output docs/benchmarks.md]

Requires: pip install contextforge[benchmark]
"""
from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
import instructor
import structlog
from datasets import load_dataset
from ragas.llms import InstructorLLM
from ragas.metrics.collections import ContextPrecision, Faithfulness
from rich.console import Console
from rich.table import Table

from contextforge import ContextEngine, Source
from contextforge.core.config import settings

log = structlog.get_logger(__name__)
console = Console()

ANTHROPIC_MODEL = "claude-sonnet-4-5"
COST_INPUT_PER_TOKEN = 0.000003     # $3.00 per 1M input tokens
COST_OUTPUT_PER_TOKEN = 0.000015    # $15.00 per 1M output tokens


@dataclass
class RunResult:
    strategy: str
    faithfulness_scores: list[float] = field(default_factory=list)
    context_precision_scores: list[float] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    latencies: list[float] = field(default_factory=list)
    utilizations: list[float] = field(default_factory=list)

    @property
    def avg_faithfulness(self) -> float:
        return sum(self.faithfulness_scores) / max(len(self.faithfulness_scores), 1)

    @property
    def avg_context_precision(self) -> float:
        return sum(self.context_precision_scores) / max(len(self.context_precision_scores), 1)

    @property
    def cost_per_1k(self) -> float:
        n = max(len(self.latencies), 1)
        per_query = (
            self.total_input_tokens * COST_INPUT_PER_TOKEN
            + self.total_output_tokens * COST_OUTPUT_PER_TOKEN
        ) / n
        return per_query * 1000

    @property
    def latency_p95(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_l = sorted(self.latencies)
        idx = int(len(sorted_l) * 0.95)
        return sorted_l[min(idx, len(sorted_l) - 1)]

    @property
    def avg_utilization(self) -> float:
        return sum(self.utilizations) / max(len(self.utilizations), 1)


def ask_llm(
    client: anthropic.Anthropic, context: str, question: str, max_retries: int = 8
) -> tuple[str, int, int]:
    for attempt in range(max_retries):
        try:
            msg = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=256,
                messages=[{
                    "role": "user",
                    "content": (
                        "Answer the question using ONLY the provided context. "
                        f"Be concise.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
                    ),
                }],
            )
            return msg.content[0].text, msg.usage.input_tokens, msg.usage.output_tokens
        except anthropic.APIStatusError as exc:
            if exc.status_code not in (429, 529) or attempt == max_retries - 1:
                raise
            wait = min(60, (2 ** attempt) + random.uniform(1, 3))
            log.warning("api_retry", status=exc.status_code, attempt=attempt + 1,
                        wait_s=round(wait, 1))
            time.sleep(wait)


def build_naive_context(titles: list[str], sentences: list[list[str]]) -> str:
    parts = []
    for title, sents in zip(titles, sentences, strict=True):
        parts.append(f"[Source: {title}]\n{' '.join(sents)}")
    return "\n\n---\n\n".join(parts)


def build_strong_baseline(
    query: str, titles: list[str], sentences: list[list[str]]
) -> str:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [" ".join(s) for s in sentences]
    q_emb = model.encode(query, normalize_embeddings=True)
    d_embs = model.encode(texts, normalize_embeddings=True)
    scores = (d_embs @ q_emb).tolist()
    top5 = sorted(zip(titles, texts, scores, strict=True), key=lambda x: x[2], reverse=True)[:5]
    return "\n\n---\n\n".join(f"[Source: {t}]\n{txt}" for t, txt, _ in top5)


def run_ragas_on_batch(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
    api_key: str,
) -> dict[str, list[float]]:
    """Run real RAGAS evaluation on a batch — direct batch_score() API (bypasses evaluate()).

    RAGAS 0.4.3 has a split-brain architecture: evaluate() type-checks against the old
    Metric base class, but collections metrics extend a different BaseMetric hierarchy.
    Calling batch_score() directly is the correct approach for collections metrics.
    """
    # AsyncAnthropic → AsyncInstructor so InstructorLLM.is_async=True → agenerate() works.
    # Pop top_p: Anthropic rejects requests with both temperature AND top_p set.
    async_anthropic = anthropic.AsyncAnthropic(api_key=api_key)
    instructor_client = instructor.from_anthropic(async_anthropic)
    llm = InstructorLLM(client=instructor_client, model=ANTHROPIC_MODEL, provider="anthropic")
    llm.model_args.pop("top_p", None)

    faith_metric = Faithfulness(llm=llm)
    prec_metric = ContextPrecision(llm=llm)

    faith_inputs = [
        {"user_input": q, "response": a, "retrieved_contexts": c}
        for q, a, c in zip(questions, answers, contexts, strict=True)
    ]
    prec_inputs = [
        {"user_input": q, "reference": gt, "retrieved_contexts": c}
        for q, gt, c in zip(questions, ground_truths, contexts, strict=True)
    ]

    # score() one at a time to respect TPM rate limits; the per-call cost is small
    faith_results = [faith_metric.score(**inp) for inp in faith_inputs]
    prec_results = [prec_metric.score(**inp) for inp in prec_inputs]

    return {
        "faithfulness": [r.value if r.value is not None else 0.0 for r in faith_results],
        "context_precision": [r.value if r.value is not None else 0.0 for r in prec_results],
    }


def run_benchmark(n_questions: int = 100) -> dict[str, RunResult]:
    console.print(
        f"\n[bold]ContextForge Benchmark — HotpotQA[/bold] ({n_questions} questions)\n"
    )
    console.print("[yellow]Using real RAGAS metrics — faithfulness + context_precision[/yellow]\n")

    dataset = load_dataset("hotpot_qa", "distractor", split="validation", streaming=True)
    questions_data = list(dataset.take(n_questions))
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key or None)
    engine = ContextEngine(token_budget=4000, top_k=20, top_n=5)
    token_budget = 4000

    results = {
        "naive": RunResult("Naive RAG (all docs)"),
        "strong": RunResult("Strong Baseline (bi-encoder top-5)"),
        "contextforge": RunResult("ContextForge"),
    }

    ragas_data: dict[str, dict[str, list]] = {
        k: {"questions": [], "answers": [], "contexts": [], "ground_truths": []}
        for k in results
    }

    for i, item in enumerate(questions_data):
        question = item["question"]
        ground_truth = item["answer"]
        titles = item["context"]["title"]
        sents = item["context"]["sentences"]
        console.print(f"  [{i + 1}/{n_questions}] {question[:70]}...")

        # Naive RAG — all docs concatenated
        naive_ctx = build_naive_context(titles, sents)
        t0 = time.time()
        naive_ans, naive_in, naive_out = ask_llm(client, naive_ctx, question)
        results["naive"].total_input_tokens += naive_in
        results["naive"].total_output_tokens += naive_out
        results["naive"].latencies.append(time.time() - t0)
        ragas_data["naive"]["questions"].append(question)
        ragas_data["naive"]["answers"].append(naive_ans)
        ragas_data["naive"]["contexts"].append([naive_ctx])
        ragas_data["naive"]["ground_truths"].append(ground_truth)

        # Strong baseline — bi-encoder top-5 only
        strong_ctx = build_strong_baseline(question, titles, sents)
        t0 = time.time()
        strong_ans, strong_in, strong_out = ask_llm(client, strong_ctx, question)
        results["strong"].total_input_tokens += strong_in
        results["strong"].total_output_tokens += strong_out
        results["strong"].latencies.append(time.time() - t0)
        ragas_data["strong"]["questions"].append(question)
        ragas_data["strong"]["answers"].append(strong_ans)
        ragas_data["strong"]["contexts"].append([strong_ctx])
        ragas_data["strong"]["ground_truths"].append(ground_truth)

        # ContextForge — full pipeline
        sources = [Source(content=" ".join(s), path=t) for t, s in zip(titles, sents, strict=True)]
        t0 = time.time()
        window = engine.build(query=question, sources=sources)
        cf_ctx = window.render()
        cf_ans, cf_in, cf_out = ask_llm(client, cf_ctx, question)
        results["contextforge"].total_input_tokens += cf_in
        results["contextforge"].total_output_tokens += cf_out
        results["contextforge"].latencies.append(time.time() - t0)
        results["contextforge"].utilizations.append(window.utilization(token_budget))
        ragas_data["contextforge"]["questions"].append(question)
        ragas_data["contextforge"]["answers"].append(cf_ans)
        ragas_data["contextforge"]["contexts"].append([cf_ctx])
        ragas_data["contextforge"]["ground_truths"].append(ground_truth)

    # Batch RAGAS evaluation for all strategies
    console.print("\n[bold]Running RAGAS evaluation...[/bold]")
    api_key = settings.anthropic_api_key
    for key, r in results.items():
        console.print(f"  Evaluating {r.strategy}...")
        d = ragas_data[key]
        scores = run_ragas_on_batch(
            d["questions"], d["answers"], d["contexts"], d["ground_truths"], api_key
        )
        r.faithfulness_scores = scores["faithfulness"]
        r.context_precision_scores = scores["context_precision"]

    return results


def print_table(results: dict[str, RunResult]) -> None:
    table = Table(title="ContextForge Benchmark Results (real RAGAS)")
    table.add_column("Strategy")
    table.add_column("RAGAS Faithfulness", justify="right")
    table.add_column("Context Precision", justify="right")
    table.add_column("Cost/1k queries", justify="right")
    table.add_column("Utilization", justify="right")
    table.add_column("Latency p95", justify="right")
    for r in results.values():
        util = f"{r.avg_utilization:.1%}" if r.utilizations else "N/A"
        table.add_row(
            r.strategy,
            f"{r.avg_faithfulness:.3f}",
            f"{r.avg_context_precision:.3f}",
            f"${r.cost_per_1k:.2f}",
            util,
            f"{r.latency_p95:.2f}s",
        )
    console.print(table)


def write_markdown(results: dict[str, RunResult], output: Path, n: int) -> None:
    lines = [
        "## Evaluation Results",
        "",
        f"**Setup:** HotpotQA distractor split, {n} questions, {ANTHROPIC_MODEL}",
        "**Metrics:** Real RAGAS faithfulness + context_precision (not a proxy)",
        "**Token counting:** tiktoken cl100k_base (approximate, ~±5% vs actual Anthropic)",
        "**Reproduce:** `uv run python scripts/benchmark.py`",
        "",
        "| Strategy | Faithfulness | Ctx Precision | Cost/1k | Utilization | Latency p95 |",
        "|----------|-------------|---------------|---------|-------------|-------------|",
    ]
    for r in results.values():
        util = f"{r.avg_utilization:.1%}" if r.utilizations else "N/A"
        lines.append(
            f"| {r.strategy} | {r.avg_faithfulness:.3f} | {r.avg_context_precision:.3f} "
            f"| ${r.cost_per_1k:.2f} | {util} | {r.latency_p95:.2f}s |"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    console.print(f"\n[green]Saved to {output}[/green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ContextForge HotpotQA benchmark")
    parser.add_argument("--n", type=int, default=100, help="Number of questions")
    parser.add_argument(
        "--output", type=Path, default=Path("docs/benchmarks.md"), help="Output markdown file"
    )
    args = parser.parse_args()
    results = run_benchmark(args.n)
    print_table(results)
    write_markdown(results, args.output, args.n)
