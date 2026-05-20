from __future__ import annotations

import re
from collections import Counter

from contextforge.core.logger import get_logger
from contextforge.models.result import RetentionScore

log = get_logger(__name__)

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "it", "this", "that", "its", "as", "into", "about", "not", "have", "has",
}


class RetentionEvaluator:
    """Measures how much key information survived compression.

    Checks: numeric values, named entities (capitalized), key terms.
    Used for library self-evaluation — not a replacement for RAGAS.
    """

    def evaluate(self, original: str, compressed: str) -> RetentionScore:
        orig_nums = set(re.findall(r"\b\d+\.?\d*\b", original))
        comp_nums = set(re.findall(r"\b\d+\.?\d*\b", compressed))
        numeric = len(orig_nums & comp_nums) / max(len(orig_nums), 1) if orig_nums else 1.0

        orig_ents = set(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", original))
        comp_ents = set(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", compressed))
        entities = len(orig_ents & comp_ents) / max(len(orig_ents), 1) if orig_ents else 1.0

        orig_words = re.findall(r"\b[a-z]{4,}\b", original.lower())
        comp_words = set(re.findall(r"\b[a-z]{4,}\b", compressed.lower()))
        freq = Counter(w for w in orig_words if w not in _STOPWORDS)
        top_terms = {w for w, _ in freq.most_common(10)}
        key_terms = len(top_terms & comp_words) / max(len(top_terms), 1) if top_terms else 1.0

        score = RetentionScore(
            key_terms_retained=key_terms,
            entities_retained=entities,
            numeric_retained=numeric,
        )
        log.debug("retention_eval", overall=score.overall)
        return score
