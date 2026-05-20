from contextforge.evaluator import RetentionEvaluator


def test_perfect_retention():
    evaluator = RetentionEvaluator()
    text = "Python 3.12 was released in 2023 with 42 new features."
    score = evaluator.evaluate(original=text, compressed=text)
    assert score.overall >= 0.99


def test_numbers_lost():
    evaluator = RetentionEvaluator()
    original = "The model achieved 94.7% accuracy on 10000 test samples."
    compressed = "The model achieved high accuracy on many samples."
    assert evaluator.evaluate(original=original, compressed=compressed).numeric_retained < 0.5


def test_entities_retained():
    evaluator = RetentionEvaluator()
    original = "Guido van Rossum created Python at the National Research Institute."
    compressed = "Guido van Rossum created Python."
    assert evaluator.evaluate(original=original, compressed=compressed).entities_retained >= 0.5


def test_score_range():
    evaluator = RetentionEvaluator()
    score = evaluator.evaluate(original="hello world", compressed="hello")
    assert 0.0 <= score.overall <= 1.0
