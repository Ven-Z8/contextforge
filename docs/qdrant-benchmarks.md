## Public Retrieval Benchmark

**Dataset:** Natural Questions dev split, 100 examples
**Candidate order:** shuffled with fixed seed `13`
**Task:** context selection over public Natural Questions long-answer candidates
**Qdrant backend:** enabled
**Evidence hit:** selected context contains a gold short answer string. This benchmark does not judge final LLM answer quality.

| Strategy | Evidence Recall | R@5 | R@10 | MRR | NDCG@10 | Avg Tokens | Tokens/Hit | Utilization | Avg Sources | Latency p50 | Latency p95 |
|----------|-----------------|-----|------|-----|---------|------------|------------|-------------|-------------|-------------|-------------|
| Shuffled candidate top-k | 0.610 | 0.610 | 0.750 | 0.400 | 0.383 | 1135 | 1861 | 28.4% | 4.9 | 0.00s | 0.00s |
| BM25 top-k | 0.770 | 0.770 | 0.870 | 0.564 | 0.525 | 1403 | 1822 | 35.1% | 4.9 | 0.00s | 0.00s |
| Vector top-k | 0.810 | 0.810 | 0.920 | 0.633 | 0.584 | 1217 | 1502 | 30.4% | 4.9 | 0.14s | 0.32s |
| Qdrant dense top-k | 0.810 | 0.810 | 0.920 | 0.633 | 0.584 | 1217 | 1502 | 30.4% | 4.9 | 0.01s | 0.01s |
| Qdrant hybrid top-k | 0.830 | 0.830 | 0.920 | 0.621 | 0.601 | 1764 | 2125 | 44.1% | 4.9 | 0.01s | 0.01s |
| Vector top-k + ContextForge | 0.810 | 0.810 | 0.810 | 0.664 | 0.516 | 727 | 898 | 18.2% | 4.9 | 0.28s | 0.55s |
| Qdrant hybrid + ContextForge | 0.810 | 0.810 | 0.810 | 0.664 | 0.515 | 765 | 945 | 19.1% | 4.9 | 0.35s | 0.71s |

### Reading This Result

Qdrant dense should track the local vector baseline because both use the same embedding model.

Qdrant hybrid measures whether dense + sparse fusion improves evidence retrieval before ContextForge compression. If hybrid recall is higher than the ContextForge row, the next engineering target is compression/reranking tuning that preserves the hybrid retriever's recall while reducing tokens.

### Current Run Notes

- Qdrant hybrid recall delta vs vector: +0.020; avg token delta: +547.
- Vector top-k + ContextForge token reduction vs vector: 40.3%; recall delta: +0.000.
- Qdrant hybrid + ContextForge recall gap vs Qdrant hybrid: +0.020. Treat this as a limitation, not a win, until compression preserves the hybrid retriever's lift.
