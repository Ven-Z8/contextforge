from __future__ import annotations

import re
from typing import Protocol

from contextforge.budget import TokenCounter
from contextforge.core.logger import get_logger
from contextforge.models.source import SourceType

log = get_logger(__name__)


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


class SentenceScorer(Protocol):
    def score(self, query: str, documents: list[str]) -> list[float]: ...


class CompressionEngine:
    """Extractive sentence-level compression for prose ONLY.

    Code and structured data are returned verbatim — no exceptions.
    This is not lossy compression: every returned token is original text.
    """

    def __init__(self, scorer: SentenceScorer | None = None) -> None:
        self._scorer = scorer

    def _get_scorer(self) -> SentenceScorer:
        if self._scorer is None:
            from contextforge.scorer import SemanticScorer

            self._scorer = SemanticScorer()
        return self._scorer

    def compress(
        self,
        text: str,
        query: str,
        target_tokens: int,
        content_type: SourceType,
        counter: TokenCounter,
    ) -> str:
        # Hard rule: no lossy compression for code or structured data
        if content_type in (SourceType.CODE, SourceType.STRUCTURED):
            log.debug("compression_skipped", reason=content_type.value)
            return text

        if counter.count(text) <= target_tokens:
            return text

        sentences = _split_sentences(text)
        if not sentences:
            return text

        scores = self._get_scorer().score(query, sentences)
        indexed = list(zip(range(len(sentences)), sentences, scores, strict=True))
        ranked = sorted(indexed, key=lambda x: x[2], reverse=True)

        selected: set[int] = set()
        used = 0
        for orig_idx, sentence, _ in ranked:
            t = counter.count(sentence)
            if used + t <= target_tokens:
                selected.add(orig_idx)
                used += t

        result = " ".join(sentences[i] for i in sorted(selected))
        log.debug("compression_done", original=counter.count(text), result=counter.count(result))
        return result
