## Public Retrieval Benchmark

**Dataset:** Natural Questions dev split, 25 examples
**Candidate order:** shuffled with fixed seed `13`
**Task:** context selection over public Natural Questions long-answer candidates
**Qdrant backend:** enabled
**Evidence hit:** selected context contains a gold short answer string. This benchmark does not judge final LLM answer quality.

| Strategy | Evidence Recall | R@5 | R@10 | MRR | NDCG@10 | Avg Tokens | Tokens/Hit | Utilization | Avg Sources | Latency p50 | Latency p95 |
|----------|-----------------|-----|------|-----|---------|------------|------------|-------------|-------------|-------------|-------------|
| Shuffled candidate top-k | 0.720 | 0.720 | 0.800 | 0.437 | 0.416 | 1267 | 1760 | 31.7% | 4.9 | 0.00s | 0.00s |
| BM25 top-k | 0.840 | 0.840 | 0.880 | 0.547 | 0.536 | 1646 | 1960 | 41.2% | 4.9 | 0.00s | 0.00s |
| Vector top-k | 0.800 | 0.800 | 0.880 | 0.616 | 0.569 | 1576 | 1970 | 39.4% | 4.9 | 0.14s | 0.28s |
| Qdrant dense top-k | 0.800 | 0.800 | 0.880 | 0.616 | 0.569 | 1576 | 1970 | 39.4% | 4.9 | 0.01s | 0.01s |
| Qdrant hybrid top-k | 0.920 | 0.920 | 0.920 | 0.598 | 0.607 | 2085 | 2266 | 52.1% | 4.9 | 0.01s | 0.01s |
| Vector top-k + ContextForge | 0.800 | 0.800 | 0.800 | 0.601 | 0.489 | 835 | 1044 | 20.9% | 4.9 | 0.24s | 0.45s |
| Qdrant hybrid + ContextForge | 0.800 | 0.800 | 0.800 | 0.601 | 0.489 | 864 | 1080 | 21.6% | 4.9 | 0.29s | 0.59s |

### Reading This Result

Qdrant dense should track the local vector baseline because both use the same embedding model.

Qdrant hybrid measures whether dense + sparse fusion improves evidence retrieval before ContextForge compression. If hybrid recall is higher than the ContextForge row, the next engineering target is compression/reranking tuning that preserves the hybrid retriever's recall while reducing tokens.