import pytest

from contextforge.reranker import CrossEncoderReranker


@pytest.fixture(scope="module")
def reranker():
    return CrossEncoderReranker()


def test_returns_top_n(reranker):
    docs = ["Python is a language.", "Cats sleep.", "Python for data science.", "Weather is cold.", "Python features."]
    results = reranker.rerank("Tell me about Python", docs, top_n=3)
    assert len(results) == 3


def test_returns_index_and_score(reranker):
    results = reranker.rerank("Python", ["Python is great.", "I like cats."], top_n=2)
    for idx, score in results:
        assert isinstance(idx, int)
        assert isinstance(score, float)


def test_most_relevant_ranks_first(reranker):
    docs = ["Unrelated: sky is blue.", "Python is a high-level programming language."]
    results = reranker.rerank("Python programming", docs, top_n=2)
    assert results[0][0] == 1  # second doc should rank first


def test_top_n_capped_at_doc_count(reranker):
    results = reranker.rerank("query", ["doc one", "doc two"], top_n=10)
    assert len(results) == 2
