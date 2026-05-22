## Evaluation Results

**Setup:** HotpotQA distractor split, 3 questions, deepseek/deepseek-v4-flash via OpenRouter
**Metrics:** Quality evaluation skipped; cost/latency/utilization only
**Token counting:** OpenRouter native usage for API cost; tiktoken cl100k_base for local budget estimates
**Reproduce:** `uv run --extra benchmark python scripts/benchmark.py --n 3 --skip-ragas --output docs/benchmarks.md`

| Strategy | Quality | Context Quality | Cost/1k | Utilization | Latency p95 |
|----------|-------------|---------------|---------|-------------|-------------|
| Naive RAG (all docs) | N/A | N/A | $0.20 | N/A | 5.15s |
| Strong Baseline (bi-encoder top-5) | N/A | N/A | $0.11 | N/A | 3.74s |
| ContextForge | N/A | N/A | $0.07 | 8.0% | 3.62s |