import pytest
from contextforge.engine import ContextEngine
from contextforge.models.source import Source

DOCS = [
    Source(content="Python is a high-level, general-purpose programming language created by Guido van Rossum.", path="intro.txt"),
    Source(content="Python 3.12 introduced performance improvements with up to 60% faster execution.", path="perf.txt"),
    Source(content="The Python Package Index hosts over 400,000 packages as of 2024.", path="eco.txt"),
    Source(content="Today's weather forecast shows rain expected throughout the afternoon.", path="noise1.txt"),
    Source(content="A recipe for chocolate chip cookies: flour, sugar, butter, chips.", path="noise2.txt"),
]


@pytest.fixture(scope="module")
def engine():
    return ContextEngine(token_budget=300, top_k=5, top_n=3)


def test_filters_noise(engine):
    window = engine.build(query="Python 3.12 performance", sources=DOCS)
    assert "Python" in window.render()
    assert "weather" not in window.render()
    assert "cookies" not in window.render()


def test_respects_budget(engine):
    window = engine.build(query="Python programming", sources=DOCS)
    assert window.token_count() <= 330


def test_attribution_preserved(engine):
    window = engine.build(query="Python history", sources=DOCS)
    assert "intro.txt" in window.render() or "perf.txt" in window.render()


def test_compression_summary(engine):
    window = engine.build(query="Python", sources=DOCS)
    summary = window.compression_summary()
    assert "total_chunks" in summary
    assert summary["total_chunks"] > 0
