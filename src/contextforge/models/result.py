from __future__ import annotations

from dataclasses import dataclass

from contextforge.models.source import Source, SourceType


@dataclass
class AssembledChunk:
    """A source after pipeline processing — carries full attribution."""

    source: Source
    compressed_content: str
    source_type: SourceType
    score: float
    original_tokens: int
    compressed_tokens: int

    @property
    def compression_ratio(self) -> float:
        if self.original_tokens == 0:
            return 1.0
        return self.compressed_tokens / self.original_tokens

    @property
    def was_compressed(self) -> bool:
        return self.compressed_tokens < self.original_tokens


@dataclass
class RetentionScore:
    key_terms_retained: float
    entities_retained: float
    numeric_retained: float

    @property
    def overall(self) -> float:
        return (self.key_terms_retained + self.entities_retained + self.numeric_retained) / 3
