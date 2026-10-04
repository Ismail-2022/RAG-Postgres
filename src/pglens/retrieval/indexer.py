"""Index chunks into Qdrant and search them.

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
from typing import Any, Protocol

from qdrant_client import QdrantClient, models

from pglens.chunking import Chunk, chunk_processed_dir
from pglens.config import INDEX_DIR, PROCESSED_DIR

COLLECTION = "pglens_docs"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # 384 dimensions, small enough for a laptop CPU

# Fixed namespace so the same chunk always maps to the same Qdrant point id
_POINT_NAMESPACE = uuid.UUID("6f1c7a0e-2b1d-4c51-9a77-3d0f2e8b5a10")


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedder:
    """Embeddings from fastembed, which runs ONNX models without PyTorch."""

    def __init__(self, model_name: str = EMBED_MODEL) -> None:
        # Imported here so the tests never load the model
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


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

    vectors = embedder.embed_documents([c.text for c in chunks])
    dimensions = len(vectors[0])

    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(
            size=dimensions, distance=models.Distance.COSINE
        ),
    )

    points = [
        models.PointStruct(
            id=str(uuid.uuid5(_POINT_NAMESPACE, chunk.chunk_id)),
            vector=vector,
            payload=_payload(chunk),
        )
        for chunk, vector in zip(chunks, vectors)
    ]
    client.upsert(collection_name=collection, points=points)
    return len(points)


def search(
    client: QdrantClient,
    embedder: Embedder,
    query: str,
    limit: int = 5,
    collection: str = COLLECTION,
) -> list[dict[str, Any]]:
    """Return the most similar chunks to the query, best first.

    Each result has the chunk's payload fields plus a "score" (cosine similarity).
    """
    response = client.query_points(
        collection_name=collection,
        query=embedder.embed_query(query),
        limit=limit,
        with_payload=True,
    )
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
