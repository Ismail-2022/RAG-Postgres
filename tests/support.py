"""Shared test doubles. No model is loaded, so tests run offline and fast."""

from __future__ import annotations

import zlib

DIMENSIONS = 64


def _slot(word: str) -> int:
    return zlib.crc32(word.encode()) % DIMENSIONS


class FakeEmbedder:
    """Deterministic bag-of-words embeddings.

    Dense: each word adds 1 to one slot. Sparse: each word is a keyword with
    weight equal to its count. Enough to check that search ranks by content.
    """

    def _dense(self, text: str) -> list[float]:
        vector = [0.0] * DIMENSIONS
        for word in text.lower().split():
            vector[_slot(word)] += 1.0
        return vector

    def _sparse(self, text: str) -> tuple[list[int], list[float]]:
        counts: dict[int, float] = {}
        for word in text.lower().split():
            counts[_slot(word)] = counts.get(_slot(word), 0.0) + 1.0
        return list(counts), list(counts.values())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._dense(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._dense(text)

    def embed_sparse_documents(
        self, texts: list[str]
    ) -> list[tuple[list[int], list[float]]]:
        return [self._sparse(t) for t in texts]

    def embed_sparse_query(self, text: str) -> tuple[list[int], list[float]]:
        return self._sparse(text)
