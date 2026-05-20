from __future__ import annotations

from contextforge.core.logger import get_logger

log = get_logger(__name__)

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Cross-encoder reranker. More accurate than bi-encoder but slower.

    Requires sentence-transformers: pip install contextforge[local]
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "sentence-transformers required. Install with: pip install contextforge[local]"
            ) from e
        log.info("loading_reranker_model", model=model_name)
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        if not documents:
            return []
        top_n = min(top_n, len(documents))
        scores = self._model.predict([(query, doc) for doc in documents]).tolist()
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        result = ranked[:top_n]
        log.debug("reranked", input=len(documents), top_n=top_n, top_score=result[0][1])
        return result
