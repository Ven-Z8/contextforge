"""ContextForge — evidence-preserving context optimizer for LLM and RAG systems.

Quick start:
    from contextforge import ContextEngine, Source

    engine = ContextEngine(token_budget=8000)
    window = engine.build(query="your question", sources=[Source(content="...")])
    llm_response = your_llm(window.render())

Install extras:
    pip install contextforge[local]      # sentence-transformers for local scoring
    pip install contextforge[benchmark]  # RAGAS + datasets for evaluation
"""
from contextforge.engine import ContextEngine
from contextforge.evaluator import RetentionEvaluator
from contextforge.models.result import AssembledChunk, RetentionScore
from contextforge.models.source import Source, SourceType
from contextforge.window import ContextWindow

__version__ = "0.1.0"
__all__ = [
    "ContextEngine",
    "Source",
    "SourceType",
    "ContextWindow",
    "RetentionScore",
    "AssembledChunk",
    "RetentionEvaluator",
]
