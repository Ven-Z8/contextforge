## Public Retrieval Benchmark

**Dataset:** Natural Questions dev split, 5 examples
**Metric:** Evidence recall = selected context contains a gold short answer string
**Note:** This benchmark evaluates public retrieval context selection, not final LLM answers.

| Strategy | Evidence Recall | Avg Tokens | Utilization | Avg Sources | Latency p95 |
|----------|-----------------|------------|-------------|-------------|-------------|
| Raw candidate top-k | 1.000 | 680 | 17.0% | 5.0 | 0.00s |
| Vector top-k | 0.800 | 1968 | 49.2% | 5.0 | 1.32s |
| Vector top-k + ContextForge | 1.000 | 1021 | 25.5% | 5.0 | 0.66s |