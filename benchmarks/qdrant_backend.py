from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from uuid import uuid4

TOKEN_RE = re.compile(r"\b\w+\b")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class SparseEncoder:
    vocab: dict[str, int]
    idf: dict[str, float]

    @classmethod
    def from_corpus(cls, texts: list[str]) -> SparseEncoder:
        vocab: dict[str, int] = {}
        doc_freq: Counter[str] = Counter()
        for text in texts:
            terms = set(tokenize(text))
            doc_freq.update(terms)
            for term in terms:
                if term not in vocab:
                    vocab[term] = len(vocab)

        doc_count = len(texts)
        idf = {
            term: math.log(1 + (doc_count - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }
        return cls(vocab=vocab, idf=idf)

    def encode(self, text: str) -> object:
        from qdrant_client import models

        counts = Counter(tokenize(text))
        weighted = [
            (self.vocab[term], count * self.idf.get(term, 0.0))
            for term, count in counts.items()
            if term in self.vocab
        ]
        weighted.sort(key=lambda item: item[0])
        return models.SparseVector(
            indices=[idx for idx, _value in weighted],
            values=[float(value) for _idx, value in weighted],
        )


class QdrantContextIndex:
    def __init__(self, contexts: list[str], dense_model: object) -> None:
        if not contexts:
            raise ValueError("Qdrant benchmark index requires at least one context")

        from qdrant_client import QdrantClient, models

        self.contexts = contexts
        self._dense_model = dense_model
        self._sparse_encoder = SparseEncoder.from_corpus(contexts)
        self._client = QdrantClient(":memory:")
        self._collection = f"contextforge_{uuid4().hex}"

        dense_vectors = self._dense_model.encode(
            contexts, normalize_embeddings=True, show_progress_bar=False
        )
        vector_size = int(dense_vectors.shape[1])
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config={
                "dense": models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )
        self._client.upsert(
            self._collection,
            points=[
                models.PointStruct(
                    id=idx,
                    vector={
                        "dense": dense_vector.tolist(),
                        "sparse": self._sparse_encoder.encode(context),
                    },
                    payload={"index": idx},
                )
                for idx, (context, dense_vector) in enumerate(
                    zip(contexts, dense_vectors, strict=True)
                )
            ],
        )

    def rank_dense(self, query: str) -> list[str]:
        query_vector = self._dense_model.encode(query, normalize_embeddings=True).tolist()
        response = self._client.query_points(
            self._collection,
            query=query_vector,
            using="dense",
            limit=len(self.contexts),
        )
        return [self.contexts[int(point.id)] for point in response.points]

    def rank_hybrid(self, query: str, prefetch_limit: int = 80) -> list[str]:
        from qdrant_client import models

        query_vector = self._dense_model.encode(query, normalize_embeddings=True).tolist()
        query_sparse = self._sparse_encoder.encode(query)
        response = self._client.query_points(
            self._collection,
            prefetch=[
                models.Prefetch(
                    query=query_vector,
                    using="dense",
                    limit=min(prefetch_limit, len(self.contexts)),
                ),
                models.Prefetch(
                    query=query_sparse,
                    using="sparse",
                    limit=min(prefetch_limit, len(self.contexts)),
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=len(self.contexts),
        )
        return [self.contexts[int(point.id)] for point in response.points]
