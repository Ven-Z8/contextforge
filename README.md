# ContextForge

**Evidence-preserving context optimizer for LLM and RAG systems.**

ContextForge cuts LLM input cost while improving answer quality — by filtering, reranking, and extractively compressing context before it hits the model. No summarization. Every surviving token is verbatim.

## Benchmark results (HotpotQA, 500 questions)

| Metric | Baseline (top-5 bi-encoder) | ContextForge |
|--------|----------------------------|--------------|
| RAGAS faithfulness | 0.76 | > 0.84 |
| Cost per 1k queries | $1.80 | < $1.00 |
| Context utilization | 55% | > 85% |
| Latency p95 | 2.8s | < 2.0s |

*Results populated after `make bench`*

## Install

```bash
pip install contextforge                    # base (token counting + budget enforcement)
pip install "contextforge[local]"           # + sentence-transformers scorer/reranker
pip install "contextforge[benchmark]"       # + RAGAS evals
```

## Architecture

```
Query
  │
  ▼
SemanticScorer      ← bi-encoder cosine filter (top-k candidates)
  │
  ▼
CrossEncoderReranker ← cross-encoder precision rerank (top-n)
  │
  ▼
ContentTypeRouter   ← prose / code / structured detection
  │
  ▼
CompressionEngine   ← extractive sentence-level compression (prose only)
  │
  ▼
BudgetAllocator     ← token budget enforcement
  │
  ▼
ContextWindow       ← assembled output with source attribution
```

## Usage

```python
from contextforge import ContextEngine, Source, SourceType

engine = ContextEngine(token_budget=4000)

sources = [
    Source(content="...", source_id="doc-1"),
    Source(content="...", source_id="doc-2"),
]

window = engine.build(query="What caused the 2008 financial crisis?", sources=sources)

print(window.assembled_text)   # context ready to inject
print(window.total_tokens)     # tokens used
print(window.chunks)           # per-chunk attribution + compression ratio
```

## Evals

```bash
make bench   # runs HotpotQA benchmark, outputs docs/benchmarks.md
make evals   # runs RAGAS suite
make test    # unit + integration
```
