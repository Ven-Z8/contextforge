from __future__ import annotations

import pytest

pytest.importorskip("qdrant_client")

from benchmarks.qdrant_backend import QdrantContextIndex, SparseEncoder  # noqa: E402


class TinyDenseModel:
    def encode(
        self,
        texts: str | list[str],
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ):
        import numpy as np

        del normalize_embeddings, show_progress_bar
        if isinstance(texts, str):
            return self._encode_one(texts)
        return np.array([self._encode_one(text) for text in texts], dtype=float)

    def _encode_one(self, text: str):
        import numpy as np

        text = text.lower()
        return np.array(
            [
                1.0 if "alpha" in text else 0.0,
                1.0 if "beta" in text else 0.0,
                1.0 if "gamma" in text else 0.0,
            ],
            dtype=float,
        )


def test_sparse_encoder_preserves_exact_terms():
    encoder = SparseEncoder.from_corpus(["alpha beta", "gamma"])
    encoded = encoder.encode("alpha missing")

    assert encoded.indices == [encoder.vocab["alpha"]]
    assert encoded.values[0] > 0


def test_qdrant_dense_and_hybrid_rank_relevant_context_first():
    contexts = [
        "gamma only unrelated",
        "alpha exact answer context",
        "beta only unrelated",
    ]
    index = QdrantContextIndex(contexts, TinyDenseModel())

    assert index.rank_dense("alpha question")[0] == "alpha exact answer context"
    assert index.rank_hybrid("alpha exact answer")[0] == "alpha exact answer context"
