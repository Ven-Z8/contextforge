# ContextForge

> **Status: Work in progress.** Code runs, but the design, API, and benchmarks are not finalized. No performance claims are being made yet.

Evidence-preserving context optimizer for LLM and RAG systems.

ContextForge filters, reranks, and extractively compresses context before it reaches the model. No summarization. Every surviving token is verbatim.

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
SemanticScorer       ← bi-encoder cosine filter (top-k candidates)
  │
  ▼
CrossEncoderReranker ← cross-encoder precision rerank (top-n)
  │
  ▼
ContentTypeRouter    ← prose / code / structured detection
  │
  ▼
CompressionEngine    ← extractive sentence-level compression (prose only)
  │
  ▼
BudgetAllocator      ← token budget enforcement
  │
  ▼
ContextWindow        ← assembled output with source attribution
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

## Commands

```bash
make bench   # runs HotpotQA benchmark, outputs docs/benchmarks.md
make evals   # runs RAGAS suite
make test    # unit + integration
```

Benchmark numbers and methodology will be published in `docs/benchmarks.md` after the suite stabilizes.

## License

TBD.
