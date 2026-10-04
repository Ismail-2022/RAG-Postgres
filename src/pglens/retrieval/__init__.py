"""Vector indexing and search over the chunked PostgreSQL docs."""

from pglens.retrieval.indexer import (
    COLLECTION,
    EMBED_MODEL,
    FastEmbedder,
    index_chunks,
    open_index,
    search,
)

__all__ = [
    "COLLECTION",
    "EMBED_MODEL",
    "FastEmbedder",
    "index_chunks",
    "open_index",
    "search",
]
