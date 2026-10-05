"""Index chunks into Qdrant and search them.

Each chunk is stored twice: a dense vector for meaning, and a sparse BM25
vector for exact keywords. Search can use either alone ("dense") or both,
with the two result lists merged by reciprocal rank fusion ("hybrid").

Uses the local (embedded) Qdrant client, which stores data in a folder and
needs no server. The same code works against a Qdrant server later by
changing the client constructor.

To build the index from data/processed/:

    python -m pglens.retrieval.indexer
"""

from __future__ import annotations

import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, Protocol

from qdrant_client import QdrantClient, models

from pglens.chunking import Chunk, chunk_processed_dir
from pglens.config import INDEX_DIR, PROCESSED_DIR

COLLECTION = "pglens_docs"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # 384 dimensions, small enough for a laptop CPU
SPARSE_MODEL = "Qdrant/bm25"  # keyword weights, no training needed
DENSE = "dense"
SPARSE = "sparse"
MODES = ("dense", "hybrid")
Mode = Literal["dense", "hybrid"]
CANDIDATES = 20  # results taken from each side before fusing in hybrid mode

# Fixed namespace so the same chunk always maps to the same Qdrant point id
_POINT_NAMESPACE = uuid.UUID("6f1c7a0e-2b1d-4c51-9a77-3d0f2e8b5a10")

SparseVector = tuple[list[int], list[float]]


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

    def embed_sparse_documents(self, texts: list[str]) -> list[SparseVector]: ...

    def embed_sparse_query(self, text: str) -> SparseVector: ...


class FastEmbedder:
    """Dense and sparse embeddings from fastembed, which runs ONNX models without PyTorch."""

    def __init__(self, model_name: str = EMBED_MODEL) -> None:
        # Imported here so the tests never load the models
        from fastembed import SparseTextEmbedding, TextEmbedding

        self._dense = TextEmbedding(model_name=model_name)
        self._sparse = SparseTextEmbedding(model_name=SPARSE_MODEL)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._dense.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._dense.query_embed(text))).tolist()

    def embed_sparse_documents(self, texts: list[str]) -> list[SparseVector]:
        return [_sparse_pair(v) for v in self._sparse.embed(texts)]

    def embed_sparse_query(self, text: str) -> SparseVector:
        return _sparse_pair(next(iter(self._sparse.query_embed(text))))


def _sparse_pair(vector: Any) -> SparseVector:
    return vector.indices.tolist(), vector.values.tolist()


def open_index(path: Path = INDEX_DIR) -> QdrantClient:
    path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path))


def index_chunks(
    client: QdrantClient,
    chunks: list[Chunk],
    embedder: Embedder,
    collection: str = COLLECTION,
) -> int:
    """Replace the collection with the given chunks. Returns the number indexed.

    The collection is rebuilt from scratch each time, so chunks removed from
    the corpus do not linger in the index.
    """
    if not chunks:
        raise ValueError("no chunks to index")

    texts = [c.text for c in chunks]
    dense = embedder.embed_documents(texts)
    sparse = embedder.embed_sparse_documents(texts)
    dimensions = len(dense[0])

    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config={
            DENSE: models.VectorParams(size=dimensions, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )

    points = [
        models.PointStruct(
            id=str(uuid.uuid5(_POINT_NAMESPACE, chunk.chunk_id)),
            vector={
                DENSE: dense_vector,
                SPARSE: models.SparseVector(indices=indices, values=values),
            },
            payload=_payload(chunk),
        )
        for chunk, dense_vector, (indices, values) in zip(chunks, dense, sparse)
    ]
    client.upsert(collection_name=collection, points=points)
    return len(points)


def search(
    client: QdrantClient,
    embedder: Embedder,
    query: str,
    limit: int = 5,
    mode: Mode = "hybrid",
    collection: str = COLLECTION,
) -> list[dict[str, Any]]:
    """Return the chunks that best match the query, best first.

    Each result has the chunk's payload fields plus a "score". In dense mode
    the score is cosine similarity; in hybrid mode it is the fused rank score.
    """
    if mode == "dense":
        response = client.query_points(
            collection_name=collection,
            query=embedder.embed_query(query),
            using=DENSE,
            limit=limit,
            with_payload=True,
        )
    elif mode == "hybrid":
        indices, values = embedder.embed_sparse_query(query)
        response = client.query_points(
            collection_name=collection,
            prefetch=[
                models.Prefetch(
                    query=embedder.embed_query(query), using=DENSE, limit=CANDIDATES
                ),
                models.Prefetch(
                    query=models.SparseVector(indices=indices, values=values),
                    using=SPARSE,
                    limit=CANDIDATES,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
    else:
        raise ValueError(f"unknown search mode: {mode!r}")

    return [
        {**(point.payload or {}), "score": point.score} for point in response.points
    ]


def _payload(chunk: Chunk) -> dict[str, Any]:
    payload = asdict(chunk)
    payload["heading_path"] = list(chunk.heading_path)
    return payload


def main() -> None:
    chunks = chunk_processed_dir(PROCESSED_DIR)
    embedder = FastEmbedder()
    client = open_index()
    count = index_chunks(client, chunks, embedder)
    print(f"indexed {count} chunks into {INDEX_DIR} (collection '{COLLECTION}')")


if __name__ == "__main__":
    main()
