"""Tests for indexing and search.

These use an in-memory Qdrant client and a small fake embedder, so they run
offline and never download the real model. The fake turns each word into a
fixed vector slot, which is enough to check that search ranks by content.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from qdrant_client import QdrantClient
from support import FakeEmbedder

from pglens.chunking import Chunk, chunk_document
from pglens.ingest.fetch import extract_main, to_markdown
from pglens.retrieval.indexer import index_chunks, search

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_FILES = sorted(FIXTURES.glob("*.html"))


def _chunk(chunk_id: str, text: str, slug: str = "page") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        slug=slug,
        title="Page Title",
        topic="test-topic",
        source_url=f"https://www.postgresql.org/docs/17/{slug}.html",
        docs_version="17",
        heading_path=("Section",),
        text=text,
    )


@pytest.fixture
def client() -> QdrantClient:
    return QdrantClient(":memory:")


def test_search_ranks_the_chunk_with_matching_words_first(client: QdrantClient) -> None:
    chunks = [
        _chunk("p#0", "vacuum reclaims dead rows left by updates", slug="maintenance"),
        _chunk("p#1", "a btree index speeds up equality lookups", slug="indexes"),
    ]
    index_chunks(client, chunks, FakeEmbedder())

    results = search(client, FakeEmbedder(), "how does vacuum reclaim rows")

    assert results[0]["chunk_id"] == "p#0"


def test_results_carry_citation_fields(client: QdrantClient) -> None:
    index_chunks(client, [_chunk("p#0", "some words here")], FakeEmbedder())

    result = search(client, FakeEmbedder(), "some words")[0]

    assert result["slug"] == "page"
    assert result["title"] == "Page Title"
    assert result["source_url"] == "https://www.postgresql.org/docs/17/page.html"
    assert result["heading_path"] == ["Section"]
    assert result["text"] == "some words here"
    assert isinstance(result["score"], float)


def test_search_respects_the_limit(client: QdrantClient) -> None:
    chunks = [_chunk(f"p#{i}", f"shared word number {i}") for i in range(10)]
    index_chunks(client, chunks, FakeEmbedder())

    assert len(search(client, FakeEmbedder(), "shared word", limit=3)) == 3


def test_reindexing_replaces_the_old_chunks(client: QdrantClient) -> None:
    embedder = FakeEmbedder()
    index_chunks(client, [_chunk(f"p#{i}", f"text {i}") for i in range(5)], embedder)

    count = index_chunks(client, [_chunk("p#0", "only one left")], embedder)

    assert count == 1
    assert client.count(collection_name="pglens_docs", exact=True).count == 1


def test_point_ids_are_stable_across_runs(client: QdrantClient) -> None:
    chunks = [_chunk("p#0", "alpha"), _chunk("p#1", "beta")]
    embedder = FakeEmbedder()

    index_chunks(client, chunks, embedder)
    first = {p.id for p in client.scroll(collection_name="pglens_docs", limit=10)[0]}
    index_chunks(client, chunks, embedder)
    second = {p.id for p in client.scroll(collection_name="pglens_docs", limit=10)[0]}

    assert first == second
    assert len(first) == 2


def test_empty_input_is_rejected(client: QdrantClient) -> None:
    with pytest.raises(ValueError):
        index_chunks(client, [], FakeEmbedder())


@pytest.mark.parametrize("fixture", FIXTURE_FILES, ids=lambda p: p.stem)
def test_saved_page_indexes_every_chunk(fixture: Path, client: QdrantClient) -> None:
    """Every chunk from a real page is indexed with its citation fields intact."""
    url = f"https://www.postgresql.org/docs/17/{fixture.stem}.html"
    title, main_html = extract_main(fixture.read_text(), url)
    body = to_markdown(main_html)
    document = (
        "---\n"
        f"title: {title!r}\n"
        f"slug: {fixture.stem}\n"
        "topic: test-topic\n"
        f"source_url: {url}\n"
        'docs_version: "17"\n'
        "---\n\n" + body
    )
    chunks = chunk_document(document)

    count = index_chunks(client, chunks, FakeEmbedder())

    assert count == len(chunks)
    results = search(client, FakeEmbedder(), chunks[0].text, limit=len(chunks))
    assert {r["chunk_id"] for r in results} == {c.chunk_id for c in chunks}
    assert all(r["source_url"] == url for r in results)


@pytest.mark.parametrize("mode", ["dense", "hybrid"])
def test_both_search_modes_return_matching_chunks(
    client: QdrantClient, mode: str
) -> None:
    chunks = [
        _chunk("p#0", "vacuum reclaims dead rows", slug="maintenance"),
        _chunk("p#1", "btree lookups use an index", slug="indexes"),
    ]
    index_chunks(client, chunks, FakeEmbedder())

    results = search(client, FakeEmbedder(), "vacuum dead rows", mode=mode)

    assert results[0]["chunk_id"] == "p#0"


def test_unknown_search_mode_is_rejected(client: QdrantClient) -> None:
    index_chunks(client, [_chunk("p#0", "some text")], FakeEmbedder())

    with pytest.raises(ValueError):
        search(client, FakeEmbedder(), "text", mode="fuzzy")  # type: ignore[arg-type]
