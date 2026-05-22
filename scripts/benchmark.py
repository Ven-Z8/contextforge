#!/usr/bin/env python3
"""ContextForge HotpotQA Benchmark — real RAGAS metrics.

Usage: uv run python scripts/benchmark.py [--n 100] [--output docs/benchmarks.md]

Requires: pip install contextforge[benchmark]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
from datasets import load_dataset
from rich.console import Console
from rich.table import Table

from contextforge import ContextEngine, Source
from contextforge.core.config import settings
from contextforge.llm import OpenRouterClient, OpenRouterError, retry_delay

log = structlog.get_logger(__name__)
console = Console()

BENCHMARK_MODEL = settings.benchmark_model
COST_INPUT_PER_TOKEN = settings.benchmark_input_cost_per_1m / 1_000_000
COST_OUTPUT_PER_TOKEN = settings.benchmark_output_cost_per_1m / 1_000_000


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


@dataclass(frozen=True)
class EvalConfig:
    mode: str
    quality_label: str
    context_label: str


def metric_cell(scores: list[float]) -> str:
    if not scores:
        return "N/A"
    return f"{sum(scores) / len(scores):.3f}"


def ask_llm(
    client: OpenRouterClient, context: str, question: str, max_retries: int = 8
) -> tuple[str, int, int]:
    for attempt in range(max_retries):
        try:
            msg = client.chat(
                model=BENCHMARK_MODEL,
                max_tokens=256,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Answer the question using ONLY the provided context. "
                            f"Be concise.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
                        ),
                    }
                ],
            )
            return msg.text, msg.input_tokens, msg.output_tokens
        except OpenRouterError as exc:
            if exc.status_code not in (429, 500, 502, 503, 504) or attempt == max_retries - 1:
                raise
            wait = retry_delay(attempt) + random.uniform(1, 3)
            log.warning("api_retry", status=exc.status_code, attempt=attempt + 1,
                        wait_s=round(wait, 1))
            time.sleep(wait)
    raise OpenRouterError("OpenRouter retry loop exhausted")


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```json").removeprefix("```").strip()
        stripped = stripped.removesuffix("```").strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise OpenRouterError(f"Expected JSON object from judge model, got: {text[:200]}")
    return stripped[start : end + 1]


def json_dumps_compact(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True)


class _OpenRouterStructuredCompletions:
    def __init__(self, client: OpenRouterClient) -> None:
        self._client = client

    async def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_model: type,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        **_: Any,
    ) -> Any:
        schema = response_model.model_json_schema()
        schema_text = json_dumps_compact(schema)
        json_instruction = {
            "role": "system",
            "content": (
                "Return only valid JSON matching the provided schema. "
                "Do not include markdown fences or commentary.\n\n"
                f"Schema:\n{schema_text}"
            ),
        }
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = await asyncio.to_thread(
                    self._client.chat,
                    model=model,
                    messages=[json_instruction, *messages],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
                return response_model.model_validate_json(_extract_json_object(response.text))
            except Exception as exc:
                last_error = exc
                if attempt == 2:
                    break
                await asyncio.sleep(retry_delay(attempt))
        raise OpenRouterError(f"Structured judge call failed: {last_error}") from last_error


class _OpenRouterStructuredChat:
    def __init__(self, client: OpenRouterClient) -> None:
        self.completions = _OpenRouterStructuredCompletions(client)


class OpenRouterStructuredClient:
    def __init__(self, client: OpenRouterClient) -> None:
        self.chat = _OpenRouterStructuredChat(client)


def build_naive_context(titles: list[str], sentences: list[list[str]]) -> str:
    parts = []
    for title, sents in zip(titles, sentences, strict=True):
        parts.append(f"[Source: {title}]\n{' '.join(sents)}")
    return "\n\n---\n\n".join(parts)


def build_strong_baseline(
    query: str, titles: list[str], sentences: list[list[str]], model: Any
) -> str:
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
    client: OpenRouterClient,
) -> dict[str, list[float]]:
    """Run real RAGAS evaluation on a batch — direct batch_score() API (bypasses evaluate()).

    RAGAS 0.4.3 has a split-brain architecture: evaluate() type-checks against the old
    Metric base class, but collections metrics extend a different BaseMetric hierarchy.
    Calling batch_score() directly is the correct approach for collections metrics.
    """
    from ragas.llms import InstructorLLM
    from ragas.metrics.collections import ContextPrecision, Faithfulness

    structured_client = OpenRouterStructuredClient(client)
    llm = InstructorLLM(
        client=structured_client,
        model=BENCHMARK_MODEL,
        provider="openrouter",
        max_tokens=1024,
        temperature=0.0,
    )

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

    # score() one at a time to respect TPM rate limits and surface progress.
    faith_results = []
    for idx, inp in enumerate(faith_inputs, start=1):
        console.print(f"    faithfulness {idx}/{len(faith_inputs)}")
        faith_results.append(faith_metric.score(**inp))

    prec_results = []
    for idx, inp in enumerate(prec_inputs, start=1):
        console.print(f"    context_precision {idx}/{len(prec_inputs)}")
        prec_results.append(prec_metric.score(**inp))

    return {
        "faithfulness": [r.value if r.value is not None else 0.0 for r in faith_results],
        "context_precision": [r.value if r.value is not None else 0.0 for r in prec_results],
    }


def normalize_for_match(text: str) -> str:
    return " ".join(text.lower().split())


def run_fast_eval_on_batch(
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict[str, list[float]]:
    """Cheap deterministic proxy metrics for iteration.

    This is not RAGAS. It answers two practical smoke-test questions:
    1. Did the answer include the gold answer string?
    2. Did the selected context include the gold answer string?
    """
    answer_contains: list[float] = []
    context_contains: list[float] = []
    for answer, ctx_list, ground_truth in zip(answers, contexts, ground_truths, strict=True):
        gt = normalize_for_match(ground_truth)
        answer_contains.append(float(gt in normalize_for_match(answer)))
        context_contains.append(float(gt in normalize_for_match("\n".join(ctx_list))))
    return {
        "quality": answer_contains,
        "context": context_contains,
    }


def run_benchmark(
    n_questions: int = 100,
    eval_config: EvalConfig | None = None,
) -> dict[str, RunResult]:
    eval_config = eval_config or EvalConfig(
        mode="ragas",
        quality_label="RAGAS Faithfulness",
        context_label="Context Precision",
    )
    console.print(
        f"\n[bold]ContextForge Benchmark — HotpotQA[/bold] ({n_questions} questions)\n"
    )
    if eval_config.mode == "ragas":
        console.print(
            "[yellow]Using real RAGAS metrics — faithfulness + context_precision[/yellow]\n"
        )
    elif eval_config.mode == "fast":
        console.print("[yellow]Using fast deterministic proxy metrics — not RAGAS[/yellow]\n")
    else:
        console.print(
            "[yellow]Skipping quality eval; measuring cost, latency, utilization[/yellow]\n"
        )

    dataset = load_dataset("hotpot_qa", "distractor", split="validation", streaming=True)
    questions_data = list(dataset.take(n_questions))
    client = OpenRouterClient()
    engine = ContextEngine(token_budget=4000, top_k=20, top_n=5)
    from sentence_transformers import SentenceTransformer

    baseline_model = SentenceTransformer("all-MiniLM-L6-v2")
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
        strong_ctx = build_strong_baseline(question, titles, sents, baseline_model)
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

    # Quality evaluation for all strategies
    if eval_config.mode == "none":
        return results

    if eval_config.mode == "ragas":
        console.print("\n[bold]Running RAGAS evaluation...[/bold]")
    else:
        console.print("\n[bold]Running fast proxy evaluation...[/bold]")

    for key, r in results.items():
        console.print(f"  Evaluating {r.strategy}...")
        d = ragas_data[key]
        if eval_config.mode == "ragas":
            scores = run_ragas_on_batch(
                d["questions"], d["answers"], d["contexts"], d["ground_truths"], client
            )
            r.faithfulness_scores = scores["faithfulness"]
            r.context_precision_scores = scores["context_precision"]
        else:
            scores = run_fast_eval_on_batch(
                d["answers"], d["contexts"], d["ground_truths"]
            )
            r.faithfulness_scores = scores["quality"]
            r.context_precision_scores = scores["context"]

    return results


def print_table(results: dict[str, RunResult], eval_config: EvalConfig) -> None:
    table = Table(title=f"ContextForge Benchmark Results ({eval_config.mode})")
    table.add_column("Strategy")
    table.add_column(eval_config.quality_label, justify="right")
    table.add_column(eval_config.context_label, justify="right")
    table.add_column("Cost/1k queries", justify="right")
    table.add_column("Utilization", justify="right")
    table.add_column("Latency p95", justify="right")
    for r in results.values():
        util = f"{r.avg_utilization:.1%}" if r.utilizations else "N/A"
        table.add_row(
            r.strategy,
            metric_cell(r.faithfulness_scores),
            metric_cell(r.context_precision_scores),
            f"${r.cost_per_1k:.2f}",
            util,
            f"{r.latency_p95:.2f}s",
        )
    console.print(table)


def write_markdown(
    results: dict[str, RunResult],
    output: Path,
    n: int,
    eval_config: EvalConfig,
    reproduce_command: str,
) -> None:
    metrics_line = {
        "ragas": "**Metrics:** Real RAGAS faithfulness + context_precision (not a proxy)",
        "fast": (
            "**Metrics:** Fast deterministic proxies: answer contains gold answer + "
            "context contains gold answer (not RAGAS)"
        ),
        "none": "**Metrics:** Quality evaluation skipped; cost/latency/utilization only",
    }[eval_config.mode]
    lines = [
        "## Evaluation Results",
        "",
        f"**Setup:** HotpotQA distractor split, {n} questions, {BENCHMARK_MODEL} via OpenRouter",
        metrics_line,
        (
            "**Token counting:** OpenRouter native usage for API cost; "
            "tiktoken cl100k_base for local budget estimates"
        ),
        f"**Reproduce:** `{reproduce_command}`",
        "",
        (
            f"| Strategy | {eval_config.quality_label} | {eval_config.context_label} "
            "| Cost/1k | Utilization | Latency p95 |"
        ),
        "|----------|-------------|---------------|---------|-------------|-------------|",
    ]
    for r in results.values():
        util = f"{r.avg_utilization:.1%}" if r.utilizations else "N/A"
        quality = metric_cell(r.faithfulness_scores)
        context = metric_cell(r.context_precision_scores)
        lines.append(
            f"| {r.strategy} | {quality} | {context} "
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
    parser.add_argument(
        "--fast-eval",
        action="store_true",
        help="Use cheap deterministic proxy metrics instead of RAGAS",
    )
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Skip quality evaluation and only report cost, latency, and utilization",
    )
    args = parser.parse_args()
    if args.fast_eval and args.skip_ragas:
        parser.error("--fast-eval and --skip-ragas are mutually exclusive")

    if args.skip_ragas:
        eval_config = EvalConfig(
            mode="none",
            quality_label="Quality",
            context_label="Context Quality",
        )
    elif args.fast_eval:
        eval_config = EvalConfig(
            mode="fast",
            quality_label="Answer Contains GT",
            context_label="Context Contains GT",
        )
    else:
        eval_config = EvalConfig(
            mode="ragas",
            quality_label="RAGAS Faithfulness",
            context_label="Context Precision",
        )

    reproduce = f"uv run --extra benchmark python scripts/benchmark.py --n {args.n}"
    if args.fast_eval:
        reproduce += " --fast-eval"
    if args.skip_ragas:
        reproduce += " --skip-ragas"
    reproduce += f" --output {args.output}"

    results = run_benchmark(args.n, eval_config)
    print_table(results, eval_config)
    write_markdown(results, args.output, args.n, eval_config, reproduce)
