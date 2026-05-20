from __future__ import annotations

import re

from contextforge.budget import TokenCounter
from contextforge.core.logger import get_logger
from contextforge.models.source import SourceType

log = get_logger(__name__)


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


class CompressionEngine:
    """Extractive sentence-level compression for prose ONLY.

    Code and structured data are returned verbatim — no exceptions.
    This is not lossy compression: every returned token is original text.
    """

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

        from contextforge.scorer import SemanticScorer

        scorer = SemanticScorer()
        sentences = _split_sentences(text)
        if not sentences:
            return text

        scores = scorer.score(query, sentences)
        ranked = sorted(zip(range(len(sentences)), sentences, scores), key=lambda x: x[2], reverse=True)

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
