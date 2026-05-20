from __future__ import annotations

from contextforge.core.logger import get_logger

log = get_logger(__name__)

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


class SemanticScorer:
    """Bi-encoder cosine similarity scorer. Fast first-pass filter.

    Requires sentence-transformers: pip install contextforge[local]
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers required. Install with: pip install contextforge[local]"
            ) from e
        log.info("loading_scorer_model", model=model_name)
        self._model = SentenceTransformer(model_name)

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        query_emb = self._model.encode(query, normalize_embeddings=True)
        doc_embs = self._model.encode(documents, normalize_embeddings=True, show_progress_bar=False)
        scores = (doc_embs @ query_emb).tolist()
        log.debug("scored_documents", count=len(documents), top_score=max(scores))
        return scores
