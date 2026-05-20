from contextforge.models.source import Source, SourceType
from contextforge.models.result import RetentionScore, AssembledChunk


def test_source_defaults():
    s = Source(content="hello world")
    assert s.content == "hello world"
    assert s.source_type == SourceType.PROSE
    assert s.path is None
    assert s.source_id is None
    assert s.metadata == {}


def test_source_with_attribution():
    s = Source(content="def foo(): pass", source_type=SourceType.CODE, path="utils.py", source_id="doc-42")
    assert s.source_type == SourceType.CODE
    assert s.source_id == "doc-42"


def test_assembled_chunk_compression_ratio():
    chunk = AssembledChunk(
        source=Source(content="hello world " * 10),
        compressed_content="hello world",
        source_type=SourceType.PROSE,
        score=0.87,
        original_tokens=20,
        compressed_tokens=2,
    )
    assert abs(chunk.compression_ratio - 0.1) < 0.01


def test_retention_score_overall():
    score = RetentionScore(key_terms_retained=0.9, entities_retained=0.8, numeric_retained=1.0)
    assert abs(score.overall - 0.9) < 0.01
