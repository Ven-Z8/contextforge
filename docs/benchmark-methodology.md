# Benchmark Methodology

## Dataset
HotpotQA distractor split (validation set)
- Multi-hop question answering
- Each question has 10 context paragraphs (2 supporting, 8 distractors)
- Ground truth answers provided

## Sample Size
Default: 100 questions
Smoke test: 10 questions (--n 10)

## Evaluation Modes
- `--skip-ragas` — iteration mode. Reports cost, latency, token usage, and utilization only.
- `--fast-eval` — deterministic proxy mode. Reports whether the answer/context contains the gold answer string. This is not RAGAS.
- default — full RAGAS mode. Runs real faithfulness + context_precision judge calls and is slower.

## Baselines
1. **Naive RAG** — all 10 context paragraphs concatenated, no filtering
2. **Strong Baseline** — bi-encoder top-5 selection only (no reranking, no compression)
3. **ContextForge** — full pipeline: bi-encoder filter → cross-encoder rerank → budget compress

## Models
- Scoring: all-MiniLM-L6-v2 (local, 22MB)
- Reranking: cross-encoder/ms-marco-MiniLM-L-6-v2 (local, 22MB)
- LLM: deepseek/deepseek-v4-flash via OpenRouter (fast, cost-effective for benchmarking)

## Metrics
- **RAGAS Faithfulness** — real RAGAS library (ragas>=0.2), not a proxy metric
- **Cost per 1k queries** — calculated from OpenRouter native prompt/completion token usage
- **Context utilization** — token_count(context) / token_budget
- **Latency p95** — wall-clock time, 95th percentile across all questions

## Token Counting
Local budget enforcement uses tiktoken cl100k_base as an approximation.
Cost calculations use OpenRouter native token counts and configured DeepSeek V4 Flash pricing.

## Known Limitations
- Token counting is approximate, not exact
- RAGAS faithfulness measures LLM answer grounding, not factual accuracy
- HotpotQA distractor split has simpler distractors than production RAG noise
- Local models (MiniLM) will produce lower scores than larger models (voyage-3, Cohere Rerank)
- Results are a lower bound — upgrading models improves all metrics
