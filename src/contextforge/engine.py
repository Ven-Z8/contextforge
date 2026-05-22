from __future__ import annotations

from contextforge.budget import BudgetAllocator, TokenCounter
from contextforge.compressor import CompressionEngine
from contextforge.core.config import settings
from contextforge.core.logger import get_logger
from contextforge.models.result import AssembledChunk
from contextforge.models.source import Source
from contextforge.reranker import CrossEncoderReranker
from contextforge.router import ContentTypeRouter
from contextforge.scorer import SemanticScorer
from contextforge.window import ContextWindow

log = get_logger(__name__)


class ContextEngine:
    """Evidence-preserving context optimizer.

    All components are injectable — swap scorers, rerankers, compressors as needed.
    Defaults use all-MiniLM-L6-v2 (scorer) and ms-marco-MiniLM-L-6-v2 (reranker).

    Pipeline:
        1. SemanticScorer        — bi-encoder filter, keeps top_k
        2. CrossEncoderReranker  — accurate rerank, keeps top_n
        3. BudgetAllocator       — distributes token_budget by relevance score
        4. ContentTypeRouter + CompressionEngine — extractive prose compression
    """

    def __init__(
        self,
        token_budget: int | None = None,
        provider: str | None = None,
        top_k: int | None = None,
        top_n: int | None = None,
        scorer: SemanticScorer | None = None,
        reranker: CrossEncoderReranker | None = None,
        compressor: CompressionEngine | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self._budget = token_budget or settings.token_budget
        self._top_k = top_k or settings.top_k
        self._top_n = top_n or settings.top_n
        self._counter = token_counter or TokenCounter(provider or settings.provider)
        self._scorer = scorer or SemanticScorer(settings.scorer_model)
        self._reranker = reranker or CrossEncoderReranker(settings.reranker_model)
        self._router = ContentTypeRouter()
        self._compressor = compressor or CompressionEngine(scorer=self._scorer)
        self._allocator = BudgetAllocator()
        log.info("engine_ready", budget=self._budget, top_k=self._top_k, top_n=self._top_n)

    def build(self, query: str, sources: list[Source]) -> ContextWindow:
        if not sources:
            return ContextWindow(query=query, chunks=[], counter=self._counter)

        # Stage 1: Semantic filter — fast, keeps top_k
        bi_scores = self._scorer.score(query, [s.content for s in sources])
        top_k = min(self._top_k, len(sources))
        paired = sorted(zip(sources, bi_scores, strict=True), key=lambda x: x[1], reverse=True)
        filtered = [s for s, _ in paired[:top_k]]
        log.debug("stage1_filter", input=len(sources), output=len(filtered))

        # Stage 2: Cross-encoder rerank — accurate, keeps top_n
        ranked = self._reranker.rerank(query, [s.content for s in filtered], self._top_n)
        reranked = [filtered[i] for i, _ in ranked]
        rerank_scores = [score for _, score in ranked]
        log.debug("stage2_rerank", output=len(reranked))

        # Stage 3: Budget allocation by relevance
        budgets = self._allocator.allocate(rerank_scores, self._budget)

        # Stage 4: Content-type routing + extractive compression
        chunks: list[AssembledChunk] = []
        for source, budget, score in zip(reranked, budgets, rerank_scores, strict=True):
            content_type = self._router.detect(source.content)
            original_tokens = self._counter.count(source.content)
            compressed = self._compressor.compress(
                source.content, query, budget, content_type, self._counter
            )
            compressed_tokens = self._counter.count(compressed)
            chunks.append(
                AssembledChunk(
                    source=source,
                    compressed_content=compressed,
                    source_type=content_type,
                    score=score,
                    original_tokens=original_tokens,
                    compressed_tokens=compressed_tokens,
                )
            )

        window = ContextWindow(query=query, chunks=chunks, counter=self._counter)
        log.info(
            "build_complete",
            sources_used=window.source_count(),
            tokens=window.token_count(),
            utilization=f"{window.utilization(self._budget):.1%}",
        )
        return window
