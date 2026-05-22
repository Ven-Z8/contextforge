from benchmarks.corpora.natural_questions import (
    examples_from_rows,
    extract_candidate_contexts,
    extract_short_answers,
    row_to_example,
)


def _row():
    return {
        "id": "nq-1",
        "question": {"text": "who proposed wave particle duality"},
        "document": {
            "title": "Wave particle duality",
            "url": "https://example.com/wiki",
            "tokens": {
                "token": [
                    "<P>",
                    "Louis",
                    "de",
                    "Broglie",
                    "proposed",
                    "that",
                    "electrons",
                    "behave",
                    "like",
                    "waves",
                    "and",
                    "particles",
                    ".",
                    "</P>",
                    "<P>",
                    "Distractor",
                    "context",
                    "without",
                    "the",
                    "answer",
                    ".",
                    "</P>",
                ],
                "is_html": [
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                ],
            },
        },
        "long_answer_candidates": {
            "start_token": [0, 14],
            "end_token": [14, 22],
            "top_level": [True, True],
        },
        "annotations": {
            "short_answers": [
                {
                    "text": ["Louis de Broglie"],
                    "start_token": [1],
                    "end_token": [4],
                }
            ]
        },
    }


def test_extract_short_answers():
    assert extract_short_answers(_row()) == ["Louis de Broglie"]


def test_extract_candidate_contexts_strips_html():
    contexts = extract_candidate_contexts(_row(), min_chars=10)
    assert contexts[0].startswith("Louis de Broglie")
    assert "<P>" not in contexts[0]


def test_row_to_example_requires_answer_in_context():
    example = row_to_example(_row(), max_contexts=5)
    assert example is not None
    assert example.question == "who proposed wave particle duality"
    assert example.answers == ["Louis de Broglie"]
    assert example.metadata["dataset"] == "natural_questions"


def test_examples_from_rows_filters_invalid_rows():
    valid = _row()
    invalid = _row()
    invalid["annotations"] = {"short_answers": [{"text": []}]}
    examples = examples_from_rows([valid, invalid])
    assert len(examples) == 1
