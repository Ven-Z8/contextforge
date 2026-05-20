## Evaluation Results

**Setup:** HotpotQA distractor split, 10 questions, claude-sonnet-4-5
**Metrics:** Real RAGAS faithfulness + context_precision (not a proxy)
**Token counting:** tiktoken cl100k_base (approximate, ~±5% vs actual Anthropic)
**Reproduce:** `uv run python scripts/benchmark.py`

| Strategy | RAGAS Faithfulness | Context Precision | Cost/1k queries | Utilization | Latency p95 |
|----------|-------------------|-------------------|-----------------|-------------|-------------|
| Naive RAG (all docs) | 0.967 | 1.000 | $5.75 | N/A | 4.62s |
| Strong Baseline (bi-encoder top-5) | 0.880 | 0.700 | $2.87 | N/A | 3.13s |
| ContextForge | 0.975 | 0.600 | $1.89 | 9.6% | 9.30s |