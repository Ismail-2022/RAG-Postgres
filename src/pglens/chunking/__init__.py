"""Heading-aware chunking of processed PostgreSQL docs."""

from pglens.chunking.chunker import (
    MAX_CHARS,
    Chunk,
    chunk_document,
    chunk_processed_dir,
    split_frontmatter,
)

__all__ = [
    "MAX_CHARS",
    "Chunk",
    "chunk_document",
    "chunk_processed_dir",
    "split_frontmatter",
]
