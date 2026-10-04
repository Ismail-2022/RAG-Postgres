"""Tests for heading-aware chunking.

Synthetic documents check each rule. The saved fixture pages are checked
against rules that every real page must satisfy, so the tests don't depend
on what any particular page says.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fetch_postgres_docs import extract_main, to_markdown
from pglens.chunking.chunker import chunk_document, split_frontmatter

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_FILES = sorted(FIXTURES.glob("*.html"))


def _doc(body: str, slug: str = "test-page", title: str = "Test Page") -> str:
    """Build a processed markdown document with frontmatter."""
    meta = {
        "title": title,
        "slug": slug,
        "topic": "test-topic",
        "source_url": f"https://www.postgresql.org/docs/17/{slug}.html",
        "docs_version": "17",
        "license": "PostgreSQL License",
        "fetched_at": "2026-01-01T00:00:00+00:00",
        "content_hash": "abc123def456",
    }
    return "---\n" + yaml.safe_dump(meta, sort_keys=False) + "---\n\n" + body


# --- frontmatter ------------------------------------------------------------


def test_frontmatter_is_parsed_and_body_returned() -> None:
    meta, body = split_frontmatter(_doc("## Heading\n\ntext\n"))

    assert meta["slug"] == "test-page"
    assert meta["docs_version"] == "17"
    assert body.startswith("## Heading")


def test_document_without_frontmatter_is_rejected() -> None:
    with pytest.raises(ValueError):
        split_frontmatter("## just markdown\n")


# --- heading splitting ------------------------------------------------------


def test_each_heading_starts_a_new_chunk() -> None:
    chunks = chunk_document(_doc("## One\n\nfirst\n\n## Two\n\nsecond\n"))

    assert [c.heading_path for c in chunks] == [("One",), ("Two",)]
    assert "first" in chunks[0].text
    assert "second" in chunks[1].text
    assert "second" not in chunks[0].text


def test_subsection_path_includes_its_parents() -> None:
    body = "## Parent\n\nintro\n\n### Child\n\ndetail\n"
    chunks = chunk_document(_doc(body))

    assert chunks[-1].heading_path == ("Parent", "Child")


def test_sibling_after_subsection_resets_the_path() -> None:
    body = "## A\n\na\n\n### A1\n\na1\n\n## B\n\nb\n"
    chunks = chunk_document(_doc(body))

    assert chunks[-1].heading_path == ("B",)


def test_text_before_first_heading_has_empty_path() -> None:
    chunks = chunk_document(_doc("Intro paragraph.\n\n## Section\n\nbody\n"))

    assert chunks[0].heading_path == ()
    assert "Intro paragraph." in chunks[0].text


def test_heading_only_section_produces_no_chunk() -> None:
    chunks = chunk_document(_doc("## Empty\n\n## Real\n\ntext\n"))

    assert [c.heading_path for c in chunks] == [("Real",)]


def test_hash_lines_inside_code_fences_are_not_headings() -> None:
    body = "## Real\n\n```\n# not a heading\n## also not\n```\n\nafter\n"
    chunks = chunk_document(_doc(body))

    assert len(chunks) == 1
    assert chunks[0].heading_path == ("Real",)
    assert "# not a heading" in chunks[0].text


# --- metadata ---------------------------------------------------------------


def test_chunks_carry_page_metadata() -> None:
    chunks = chunk_document(_doc("## S\n\ntext\n", slug="my-page", title="My Page"))
    chunk = chunks[0]

    assert chunk.slug == "my-page"
    assert chunk.title == "My Page"
    assert chunk.topic == "test-topic"
    assert chunk.docs_version == "17"
    assert chunk.source_url == "https://www.postgresql.org/docs/17/my-page.html"


def test_chunk_ids_are_unique_and_numbered_per_page() -> None:
    body = "## A\n\na\n\n## B\n\nb\n\n## C\n\nc\n"
    chunks = chunk_document(_doc(body, slug="pg"))

    assert [c.chunk_id for c in chunks] == ["pg#0", "pg#1", "pg#2"]


def test_every_piece_repeats_its_heading() -> None:
    paragraph = "word " * 200
    body = "## Long\n\n" + "\n\n".join([paragraph] * 6) + "\n"
    chunks = chunk_document(_doc(body), max_chars=1000)

    assert len(chunks) > 1
    assert all(c.text.startswith("## Long") for c in chunks)


# --- size limit -------------------------------------------------------------


def test_long_section_is_split_into_pieces_under_the_limit() -> None:
    paragraph = "word " * 100
    body = "## Long\n\n" + "\n\n".join([paragraph] * 10) + "\n"
    chunks = chunk_document(_doc(body), max_chars=1200)

    assert len(chunks) > 1
    assert all(len(c.text) <= 1200 for c in chunks)


def test_splitting_never_loses_or_duplicates_paragraphs() -> None:
    paragraphs = [f"Paragraph number {i} has some words." for i in range(30)]
    body = "## Long\n\n" + "\n\n".join(paragraphs) + "\n"
    chunks = chunk_document(_doc(body), max_chars=300)

    rejoined = "\n".join(c.text for c in chunks)
    for paragraph in paragraphs:
        assert rejoined.count(paragraph) == 1


def test_short_section_stays_one_chunk() -> None:
    chunks = chunk_document(_doc("## Short\n\nsmall text\n"), max_chars=1200)

    assert len(chunks) == 1


def test_code_block_is_never_split_even_when_oversized() -> None:
    code = "\n".join(f"line {i}" for i in range(400))
    body = f"## Code\n\nintro\n\n```\n{code}\n```\n\nafter\n"
    chunks = chunk_document(_doc(body), max_chars=500)

    code_chunks = [c for c in chunks if "line 0" in c.text]
    assert len(code_chunks) == 1
    assert "line 399" in code_chunks[0].text
    assert code_chunks[0].text.count("```") == 2


def test_table_is_never_split_even_when_oversized() -> None:
    rows = "\n".join(f"| row {i} | value {i} |" for i in range(200))
    body = f"## Table\n\n| a | b |\n| --- | --- |\n{rows}\n\nafter\n"
    chunks = chunk_document(_doc(body), max_chars=500)

    table_chunks = [c for c in chunks if "row 0 " in c.text]
    assert len(table_chunks) == 1
    assert "row 199 " in table_chunks[0].text


# --- real page checks (run on every saved fixture) --------------------------


@pytest.mark.parametrize("fixture", FIXTURE_FILES, ids=lambda p: p.stem)
def test_saved_page_chunks_cleanly(fixture: Path) -> None:
    """Rules every real page must satisfy, whatever the page is about.

    The page goes through the same conversion as the live pipeline first,
    so this also checks that ingestion output is chunkable.
    """
    url = f"https://www.postgresql.org/docs/17/{fixture.stem}.html"
    title, main_html = extract_main(fixture.read_text(), url)
    document = _doc(to_markdown(main_html), slug=fixture.stem, title=title)

    chunks = chunk_document(document)

    assert chunks
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    for chunk in chunks:
        assert chunk.text.strip()
        assert chunk.text.count("```") % 2 == 0
        assert chunk.source_url == url
        assert chunk.title == title
