from __future__ import annotations

from dataclasses import dataclass

from contextforge.budget import TokenCounter
from contextforge.models.result import AssembledChunk


@dataclass
class ContextWindow:
    """Assembled, budget-aware context with full source attribution.

    Every token is verbatim original text — no hallucination risk from compression.
    """

    query: str
    chunks: list[AssembledChunk]
    counter: TokenCounter

    def render(self) -> str:
        """Render context as string for LLM injection with source labels."""
        if not self.chunks:
            return ""
        parts = []
        for chunk in self.chunks:
            source = chunk.source
            label = f"[Source: {source.path}]" if source.path else "[Source]"
            if source.source_id:
                label = f"[Source: {source.path or source.source_id} | id={source.source_id}]"
            parts.append(f"{label}\n{chunk.compressed_content}")
        return "\n\n---\n\n".join(parts)

    def token_count(self) -> int:
        return self.counter.count(self.render())

    def utilization(self, total_budget: int) -> float:
        return self.token_count() / total_budget if total_budget > 0 else 0.0

    def source_count(self) -> int:
        return len(self.chunks)

    def compression_summary(self) -> dict:
        """Returns metadata about what was compressed and by how much."""
        compressed = [c for c in self.chunks if c.was_compressed]
        ratios = [c.compression_ratio for c in self.chunks]
        return {
            "total_chunks": len(self.chunks),
            "chunks_compressed": len(compressed),
            "avg_compression_ratio": sum(ratios) / len(ratios) if ratios else 1.0,
            "original_tokens": sum(c.original_tokens for c in self.chunks),
            "final_tokens": sum(c.compressed_tokens for c in self.chunks),
        }
