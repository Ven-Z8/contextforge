# Benchmark Methodology

## Dataset
HotpotQA distractor split (validation set)
- Multi-hop question answering
- Each question has 10 context paragraphs (2 supporting, 8 distractors)
- Ground truth answers provided

Natural Questions dev split
- Public Wikipedia question-answering corpus
- Uses document tokens and top-level long-answer candidates as retrieval candidates
- Evaluates whether selected context contains a gold short answer string
- Shuffles candidate order with a fixed seed so source ordering is not a hidden baseline advantage

## Sample Size
Default public retrieval benchmark: 100 questions
Smoke test: 5-10 questions (`--n 5` or `--n 10`)

## Evaluation Modes
- `--skip-ragas` — iteration mode. Reports cost, latency, token usage, and utilization only.
- `--fast-eval` — deterministic proxy mode. Reports whether the answer/context contains the gold answer string. This is not RAGAS.
- default — full RAGAS mode. Runs real faithfulness + context_precision judge calls and is slower.

## Baselines
1. **Naive RAG** — all 10 context paragraphs concatenated, no filtering
2. **Strong Baseline** — bi-encoder top-5 selection only (no reranking, no compression)
3. **ContextForge** — full pipeline: bi-encoder filter → cross-encoder rerank → budget compress

## Public Retrieval Benchmark
`benchmarks/eval.py` runs public context-selection evals against Natural Questions:

```bash
uv run --extra benchmark python benchmarks/eval.py --dataset natural_questions --n 100
```

Qdrant comparison mode:

```bash
uv run --extra benchmark --extra qdrant python benchmarks/eval.py \
  --dataset natural_questions --n 100 --include-qdrant \
  --output docs/qdrant-benchmarks.md \
  --failure-output docs/qdrant-failure-analysis.md \
  --gates-output docs/qdrant-gates.md
```

Strategies:
1. **Shuffled candidate top-k** — fixed-seed shuffled public candidates, no scoring
2. **BM25 top-k** — lexical sparse baseline
3. **Vector top-k** — local MiniLM semantic scoring over candidates
4. **Qdrant dense top-k** — in-memory Qdrant dense vector search
5. **Qdrant hybrid top-k** — Qdrant dense + sparse prefetch with RRF fusion
6. **Vector top-k + ContextForge** — vector retrieval candidates optimized by ContextForge
7. **Qdrant hybrid + ContextForge** — Qdrant hybrid candidates optimized by ContextForge

Qdrant mode uses an in-memory per-example collection so it requires no Docker server. Reported
Qdrant latencies measure query-time ranking; per-example index construction is excluded because
production retrieval systems index before serving queries.

Qdrant mode also writes:
- `docs/qdrant-failure-analysis.md` — examples where evidence is lost after ContextForge or a
  ContextForge output exceeds the token budget
- `docs/qdrant-gates.md` — pass/fail gates for recall preservation, token reduction, and budget
  compliance

Failed gates block broad benchmark claims. They are still useful evidence and should be reported
as limitations instead of hidden.

## Models
- Scoring: all-MiniLM-L6-v2 (local, 22MB)
- Reranking: cross-encoder/ms-marco-MiniLM-L-6-v2 (local, 22MB)
- LLM: deepseek/deepseek-v4-flash via OpenRouter (fast, cost-effective for benchmarking)

## Metrics
- **RAGAS Faithfulness** — real RAGAS library (ragas>=0.2), not a proxy metric
- **Cost per 1k queries** — calculated from OpenRouter native prompt/completion token usage
- **Context utilization** — token_count(context) / token_budget
- **Evidence recall** — selected context contains at least one gold short answer string
- **Recall@5 / Recall@10** — answer evidence appears within top 5 or top 10 ranked candidates
- **MRR** — reciprocal rank of first answer-bearing candidate
- **NDCG@10** — ranking quality for answer-bearing candidates at depth 10
- **Tokens per evidence hit** — total selected context tokens divided by evidence hits
- **Latency p50 / p95** — wall-clock time across all questions

## Token Counting
Local budget enforcement uses tiktoken cl100k_base as an approximation.
Cost calculations use OpenRouter native token counts and configured DeepSeek V4 Flash pricing.

## Known Limitations
- Token counting is approximate, not exact
- RAGAS faithfulness measures LLM answer grounding, not factual accuracy
- HotpotQA distractor split has simpler distractors than production RAG noise
- Local models (MiniLM) will produce lower scores than larger models (voyage-3, Cohere Rerank)
- Results are a lower bound — upgrading models improves all metrics
- Natural Questions benchmark is still context selection over provided public candidates, not full corpus retrieval
- Gold evidence is approximated by short-answer string containment, so it is useful but not a substitute for qrels
