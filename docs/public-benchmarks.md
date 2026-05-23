## Public Retrieval Benchmark

**Dataset:** Natural Questions dev split, 100 examples
**Candidate order:** shuffled with fixed seed `13`
**Task:** context selection over public Natural Questions long-answer candidates
**Evidence hit:** selected context contains a gold short answer string. This benchmark does not judge final LLM answer quality.

| Strategy | Evidence Recall | R@5 | R@10 | MRR | NDCG@10 | Avg Tokens | Tokens/Hit | Utilization | Avg Sources | Latency p50 | Latency p95 |
|----------|-----------------|-----|------|-----|---------|------------|------------|-------------|-------------|-------------|-------------|
| Shuffled candidate top-k | 0.610 | 0.610 | 0.750 | 0.400 | 0.383 | 1135 | 1861 | 28.4% | 4.9 | 0.00s | 0.00s |
| BM25 top-k | 0.770 | 0.770 | 0.870 | 0.564 | 0.525 | 1403 | 1822 | 35.1% | 4.9 | 0.00s | 0.00s |
| Vector top-k | 0.810 | 0.810 | 0.920 | 0.633 | 0.584 | 1217 | 1502 | 30.4% | 4.9 | 0.14s | 0.28s |
| Vector top-k + ContextForge | 0.810 | 0.810 | 0.810 | 0.664 | 0.516 | 727 | 898 | 18.2% | 4.9 | 0.29s | 0.57s |