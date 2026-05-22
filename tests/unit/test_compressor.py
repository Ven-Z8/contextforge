from contextforge.budget import TokenCounter
from contextforge.compressor import CompressionEngine
from contextforge.models.source import SourceType


class FakeScorer:
    def __init__(self):
        self.calls = 0

    def score(self, _query, documents):
        self.calls += 1
        return [float(i) for i, _ in enumerate(documents)]


def test_prose_within_budget_unchanged():
    engine = CompressionEngine()
    counter = TokenCounter()
    result = engine.compress("Short text.", "query", target_tokens=100, content_type=SourceType.PROSE, counter=counter)
    assert result == "Short text."


def test_code_never_lossy_compressed():
    engine = CompressionEngine()
    counter = TokenCounter()
    code = "def foo():\n    " + "x = 1\n    " * 200
    result = engine.compress(code, "query", target_tokens=10, content_type=SourceType.CODE, counter=counter)
    assert result == code  # verbatim regardless of budget


def test_structured_never_lossy_compressed():
    engine = CompressionEngine()
    counter = TokenCounter()
    data = '{"key": "' + "value " * 200 + '"}'
    result = engine.compress(data, "query", target_tokens=10, content_type=SourceType.STRUCTURED, counter=counter)
    assert result == data


def test_prose_reduces_tokens():
    engine = CompressionEngine()
    counter = TokenCounter()
    text = " ".join([f"Sentence {i} about topic {i}." for i in range(50)])
    result = engine.compress(text, "topic 1", target_tokens=100, content_type=SourceType.PROSE, counter=counter)
    assert counter.count(result) <= 110  # within tolerance


def test_prose_retains_relevant_sentences():
    engine = CompressionEngine()
    counter = TokenCounter()
    text = (
        "The sky is blue and clouds are white. "
        "Python is a high-level programming language created by Guido van Rossum. "
        "Cats enjoy sleeping in warm places. "
        "Python is widely used for machine learning."
    )
    result = engine.compress(text, "Python programming", target_tokens=40, content_type=SourceType.PROSE, counter=counter)
    assert "Python" in result


def test_reuses_injected_scorer_for_compression():
    scorer = FakeScorer()
    engine = CompressionEngine(scorer=scorer)
    counter = TokenCounter()
    text = (
        "First sentence has low score. "
        "Second sentence has medium score. "
        "Third sentence has the highest score."
    )

    result = engine.compress(
        text,
        "highest score",
        target_tokens=12,
        content_type=SourceType.PROSE,
        counter=counter,
    )

    assert scorer.calls == 1
    assert "Third sentence" in result
