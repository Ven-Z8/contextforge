# ContextForge — Claude Code Instructions

## What this is
Evidence-preserving context optimizer. pip install contextforge.
Pipeline: semantic filter → cross-encoder rerank → budget allocate → extractive compress.

## Build goal
Achieve on HotpotQA vs strong baseline (bi-encoder top-5):
- RAGAS faithfulness > 0.84 (real RAGAS, not proxy)
- Cost per 1k queries < $1.00
- Context utilization > 85%

## Hard rules
- NEVER lossy-compress code, JSON, YAML, SQL, markdown tables
- NEVER claim exact Anthropic token counting — we use tiktoken cl100k_base (approximate, ~±5%)
- NEVER call any metric RAGAS unless real ragas library is used
- structlog everywhere, never print()
- Full type hints on every function signature
- ContextEngine must accept injectable components

## Commands
```bash
uv run pytest tests/ -v                         # all tests
uv run pytest tests/unit -v                     # unit only
uv run python scripts/benchmark.py --n 10       # quick smoke benchmark
uv run ruff check src/ tests/                   # lint
```

## Package structure
src/contextforge/
├── engine.py       ← ContextEngine (composable pipeline)
├── scorer.py       ← SemanticScorer (bi-encoder filter)
├── reranker.py     ← CrossEncoderReranker
├── budget.py       ← BudgetAllocator + TokenCounter
├── router.py       ← ContentTypeRouter
├── compressor.py   ← CompressionEngine (extractive only)
├── window.py       ← ContextWindow (assembled output)
├── evaluator.py    ← RetentionEvaluator
├── cli.py          ← typer CLI
├── core/           ← config, logger
└── models/         ← source.py, result.py
