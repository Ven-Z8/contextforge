import pytest
from contextforge.engine import ContextEngine
from contextforge.models.source import Source, SourceType
from contextforge.window import ContextWindow


@pytest.fixture(scope="module")
def engine():
    return ContextEngine(token_budget=500, top_k=10, top_n=3)


def test_returns_context_window(engine):
    sources = [
        Source(content="Python is a high-level programming language.", path="doc1.txt"),
        Source(content="The weather is sunny today.", path="doc2.txt"),
    ]
    window = engine.build(query="Tell me about Python", sources=sources)
    assert isinstance(window, ContextWindow)


def test_filters_irrelevant_sources(engine):
    sources = [
        Source(content="Python is great for data science."),
        Source(content="Cats enjoy sleeping in warm spots."),
        Source(content="Python has a rich ecosystem."),
    ]
    window = engine.build(query="Python programming", sources=sources)
    assert "Python" in window.render()


def test_respects_token_budget(engine):
    sources = [Source(content="word " * 300, path=f"doc{i}.txt") for i in range(10)]
    window = engine.build(query="test query", sources=sources)
    assert window.token_count() <= 550  # budget + separator tolerance


def test_empty_sources(engine):
    window = engine.build(query="test", sources=[])
    assert window.source_count() == 0


def test_never_compresses_code(engine):
    code = "def solve(n):\n    " + "return n * 2\n" * 50
    sources = [Source(content=code, source_type=SourceType.CODE, path="solver.py")]
    window = engine.build(query="How does solve work?", sources=sources)
    assert "def solve" in window.render()


def test_injectable_scorer():
    """ContextEngine accepts custom scorer."""
    from contextforge.scorer import SemanticScorer
    custom_scorer = SemanticScorer(model_name="all-MiniLM-L6-v2")
    engine = ContextEngine(token_budget=500, top_k=5, top_n=2, scorer=custom_scorer)
    sources = [Source(content="Python is a language."), Source(content="Cats are nice.")]
    window = engine.build(query="Python", sources=sources)
    assert isinstance(window, ContextWindow)
