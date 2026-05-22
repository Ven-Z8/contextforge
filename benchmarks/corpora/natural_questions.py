from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from benchmarks.schemas import BenchmarkExample


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _tokens_to_text(tokens: dict[str, list[Any]], start: int, end: int) -> str:
    words: list[str] = []
    token_values = tokens["token"]
    html_flags = tokens.get("is_html") or [False] * len(token_values)
    for token, is_html in zip(token_values[start:end], html_flags[start:end], strict=False):
        if is_html:
            continue
        token = str(token).strip()
        if not token or token.startswith("<"):
            continue
        words.append(token)
    return " ".join(words).strip()


def extract_short_answers(row: dict[str, Any]) -> list[str]:
    answers: list[str] = []
    for item in row.get("annotations", {}).get("short_answers", []):
        for text in item.get("text", []):
            text = str(text).strip()
            if text:
                answers.append(text)
    return list(dict.fromkeys(answers))


def extract_candidate_contexts(
    row: dict[str, Any],
    max_contexts: int = 60,
    min_chars: int = 40,
) -> list[str]:
    tokens = row["document"]["tokens"]
    candidates = row["long_answer_candidates"]
    starts = candidates["start_token"]
    ends = candidates["end_token"]
    top_level = candidates.get("top_level") or [True] * len(starts)

    contexts: list[str] = []
    seen: set[str] = set()
    for start, end, is_top_level in zip(starts, ends, top_level, strict=False):
        if not is_top_level:
            continue
        text = _tokens_to_text(tokens, int(start), int(end))
        if len(text) < min_chars:
            continue
        key = _normalize(text)
        if key in seen:
            continue
        seen.add(key)
        contexts.append(text)
        if len(contexts) >= max_contexts:
            break
    return contexts


def row_to_example(row: dict[str, Any], max_contexts: int = 60) -> BenchmarkExample | None:
    answers = extract_short_answers(row)
    if not answers:
        return None

    contexts = extract_candidate_contexts(row, max_contexts=max_contexts)
    if not contexts:
        return None

    joined_context = _normalize("\n".join(contexts))
    if not any(_normalize(answer) in joined_context for answer in answers):
        return None

    return BenchmarkExample(
        id=str(row["id"]),
        question=str(row["question"]["text"]),
        answers=answers,
        contexts=contexts,
        metadata={
            "dataset": "natural_questions",
            "title": str(row["document"].get("title") or ""),
            "url": str(row["document"].get("url") or ""),
        },
    )


def iter_natural_questions(
    n: int,
    split: str = "validation",
    config: str = "dev",
    max_contexts: int = 60,
) -> Iterator[BenchmarkExample]:
    from datasets import load_dataset

    dataset = load_dataset("natural_questions", config, split=split, streaming=True)
    yielded = 0
    for row in dataset:
        example = row_to_example(row, max_contexts=max_contexts)
        if example is None:
            continue
        yield example
        yielded += 1
        if yielded >= n:
            break


def examples_from_rows(
    rows: Iterable[dict[str, Any]],
    max_contexts: int = 60,
) -> list[BenchmarkExample]:
    examples: list[BenchmarkExample] = []
    for row in rows:
        example = row_to_example(row, max_contexts=max_contexts)
        if example is not None:
            examples.append(example)
    return examples
