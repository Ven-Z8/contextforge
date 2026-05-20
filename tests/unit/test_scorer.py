import pytest
from contextforge.scorer import SemanticScorer


@pytest.fixture(scope="module")
def scorer():
    return SemanticScorer()


def test_returns_score_per_doc(scorer):
    scores = scorer.score("What is Python?", ["Python is a language.", "Cats are pets."])
    assert len(scores) == 2


def test_relevant_doc_scores_higher(scorer):
    scores = scorer.score(
        "What is Python?",
        ["Python is a high-level programming language.", "The weather is sunny today."],
    )
    assert scores[0] > scores[1]


def test_empty_docs(scorer):
    assert scorer.score("query", []) == []


def test_scores_between_neg1_and_1(scorer):
    scores = scorer.score("machine learning", ["deep learning", "cooking pasta"])
    assert all(-1.0 <= s <= 1.0 for s in scores)
