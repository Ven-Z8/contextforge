from contextforge.window import ContextWindow
from contextforge.models.source import Source, SourceType
from contextforge.models.result import AssembledChunk
from contextforge.budget import TokenCounter


def _make_chunk(content: str, compressed: str, score: float = 0.9) -> AssembledChunk:
    counter = TokenCounter()
    return AssembledChunk(
        source=Source(content=content, path="doc.txt"),
        compressed_content=compressed,
        source_type=SourceType.PROSE,
        score=score,
        original_tokens=counter.count(content),
        compressed_tokens=counter.count(compressed),
    )


def test_render_contains_content():
    counter = TokenCounter()
    chunk = _make_chunk("hello world", "hello world")
    window = ContextWindow(query="test", chunks=[chunk], counter=counter)
    assert "hello world" in window.render()


def test_render_includes_source_path():
    counter = TokenCounter()
    chunk = _make_chunk("content here", "content here")
    window = ContextWindow(query="test", chunks=[chunk], counter=counter)
    assert "doc.txt" in window.render()


def test_token_count():
    counter = TokenCounter()
    chunk = _make_chunk("hello world", "hello world")
    window = ContextWindow(query="test", chunks=[chunk], counter=counter)
    assert window.token_count() > 0


def test_utilization():
    counter = TokenCounter()
    chunk = _make_chunk("hello world", "hello world")
    window = ContextWindow(query="test", chunks=[chunk], counter=counter)
    assert 0.0 < window.utilization(8000) < 1.0


def test_compression_summary():
    counter = TokenCounter()
    chunk = _make_chunk("word " * 100, "word " * 20)
    window = ContextWindow(query="test", chunks=[chunk], counter=counter)
    summary = window.compression_summary()
    assert summary["chunks_compressed"] == 1
    assert summary["avg_compression_ratio"] < 1.0
