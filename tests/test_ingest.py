"""Offline tests for the PostgreSQL docs ingestion script.

None of these tests touch the network, and none depend on what a specific
page says. They check the rules the converter must follow, so they keep
working when sources.yaml changes.

To add a real-page regression case, save a raw page into tests/fixtures/
as <slug>.html. Every fixture is picked up automatically.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urljoin

import httpx
import pytest
import yaml

from pglens.ingest import fetch as fpd

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_FILES = sorted(FIXTURES.glob("*.html"))
BASE_URL = "https://www.postgresql.org/docs/17"
PAGE_URL = f"{BASE_URL}/example-page.html"

# A markdown link whose target is not an absolute http(s) URL
RELATIVE_LINK = re.compile(r"\]\((?!https?://)[^)]*\)")
# A heading permalink as markdownify renders it, e.g. [#](#SOME-ID)
PERMALINK = re.compile(r"\[#\]\(#")


def _page(body: str, title: str = "1.2.&nbsp;Example Heading") -> str:
    """Wrap a body fragment in the same structure the docs site uses."""
    return (
        "<html><body>"
        f'<h2>{title} <a href="#EXAMPLE-ID" class="id_link">#</a></h2>'
        f'<div id="docContent">{body}</div>'
        "</body></html>"
    )


# --- extract_main -----------------------------------------------------------


def test_title_drops_permalink_and_normalises_nbsp() -> None:
    title, _ = fpd.extract_main(_page("<p>text</p>"), PAGE_URL)

    assert title == "1.2. Example Heading"


def test_heading_permalinks_are_removed_from_content() -> None:
    _, content = fpd.extract_main(_page("<p>text</p>"), PAGE_URL)

    assert "id_link" not in content
    assert 'href="#EXAMPLE-ID"' not in content


def test_relative_links_become_absolute() -> None:
    body = '<p><a href="some-page.html">link</a></p>'
    _, content = fpd.extract_main(_page(body), PAGE_URL)

    assert f'href="{urljoin(PAGE_URL, "some-page.html")}"' in content


def test_relative_link_with_fragment_keeps_the_fragment() -> None:
    body = '<p><a href="some-page.html#SECTION">link</a></p>'
    _, content = fpd.extract_main(_page(body), PAGE_URL)

    assert f'href="{urljoin(PAGE_URL, "some-page.html#SECTION")}"' in content


def test_in_page_anchor_points_at_the_section() -> None:
    body = '<p><a href="#SECTION">section</a></p>'
    _, content = fpd.extract_main(_page(body), PAGE_URL)

    assert f'href="{PAGE_URL}#SECTION"' in content


def test_navigation_chrome_is_removed() -> None:
    html = (
        "<html><body>"
        '<div class="navheader">NAV-HEADER-TEXT</div>'
        '<div id="docContent"><p>KEEP-ME</p></div>'
        '<div class="navfooter">NAV-FOOTER-TEXT</div>'
        "</body></html>"
    )
    _, content = fpd.extract_main(html, PAGE_URL)

    assert "KEEP-ME" in content
    assert "NAV-HEADER-TEXT" not in content
    assert "NAV-FOOTER-TEXT" not in content


def test_missing_docContent_falls_back_to_body() -> None:
    html = "<html><body><h1>Title</h1><p>body text</p></body></html>"
    title, content = fpd.extract_main(html, PAGE_URL)

    assert title == "Title"
    assert "body text" in content


def test_page_without_headings_gets_untitled() -> None:
    html = "<html><body><p>no headings here</p></body></html>"
    title, _ = fpd.extract_main(html, PAGE_URL)

    assert title == "Untitled"


# --- to_markdown ------------------------------------------------------------


def test_markdown_uses_atx_headings() -> None:
    md = fpd.to_markdown("<h2>Heading</h2><p>para</p>")

    assert md.startswith("## Heading")


def test_markdown_has_no_non_breaking_spaces() -> None:
    md = fpd.to_markdown("<p>Section&nbsp;52.12 and&nbsp;more</p>")

    assert "\xa0" not in md
    assert "Section 52.12 and more" in md


def test_markdown_collapses_runs_of_blank_lines() -> None:
    md = fpd.to_markdown("<p>one</p><p></p><p></p><p></p><p>two</p>")

    assert "\n\n\n" not in md


def test_code_blocks_keep_balanced_fences() -> None:
    md = fpd.to_markdown("<pre>SELECT 1;</pre><p>after</p>")

    assert md.count("```") % 2 == 0
    assert "SELECT 1;" in md


# --- real page checks (run on every saved fixture) --------------------------


@pytest.mark.parametrize("fixture", FIXTURE_FILES, ids=lambda p: p.stem)
def test_saved_page_converts_cleanly(fixture: Path) -> None:
    """Rules every real docs page must satisfy, whatever the page is about.

    These are the bugs found in the first live run: permalinks, relative
    links, non-breaking spaces and unbalanced code fences.
    """
    url = f"{BASE_URL}/{fixture.stem}.html"
    title, main_html = fpd.extract_main(fixture.read_text(), url)
    body = fpd.to_markdown(main_html)

    assert title and title != "Untitled"
    assert not title.endswith("#")
    assert "\xa0" not in title and "\xa0" not in body
    assert not PERMALINK.search(body)
    assert not RELATIVE_LINK.search(body)
    assert body.count("```") % 2 == 0
    assert len(body) > 100


# --- run() end to end, offline ----------------------------------------------


def _write_sources(path: Path, pages: list[dict[str, str]]) -> None:
    config = {
        "version": "17",
        "base_url": BASE_URL,
        "license": "PostgreSQL License",
        "pages": pages,
    }
    path.write_text(yaml.safe_dump(config))


def _read_frontmatter(path: Path) -> dict[str, object]:
    _, fm, _ = path.read_text().split("---\n", 2)
    loaded: dict[str, object] = yaml.safe_load(fm)
    return loaded


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    """A temporary project folder, so run() never writes to the real data/."""
    return tmp_path


def _run(workdir: Path) -> None:
    """Run the ingestion against the temporary folder's sources, raw and processed dirs."""
    fpd.run(
        sources=workdir / "sources.yaml",
        raw_dir=workdir / "raw",
        processed_dir=workdir / "processed",
    )


@pytest.mark.parametrize("fixture", FIXTURE_FILES, ids=lambda p: p.stem)
def test_run_writes_processed_file_with_frontmatter(
    fixture: Path, workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    slug = fixture.stem
    html = fixture.read_text()
    monkeypatch.setattr(fpd, "fetch_html", lambda url: html)
    _write_sources(workdir / "sources.yaml", [{"slug": slug, "topic": "test-topic"}])

    _run(workdir)

    meta = _read_frontmatter(workdir / "processed" / f"{slug}.md")
    assert meta["slug"] == slug
    assert meta["topic"] == "test-topic"
    assert meta["source_url"] == f"{BASE_URL}/{slug}.html"
    assert meta["docs_version"] == "17"
    assert meta["title"]
    assert len(str(meta["content_hash"])) == 12
    assert (workdir / "raw" / f"{slug}.html").read_text() == html


def test_run_escapes_quotes_in_title(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tricky = 'Tuning: "VACUUM" and # comments'
    html = _page("<p>body</p>", title=tricky)
    monkeypatch.setattr(fpd, "fetch_html", lambda url: html)
    _write_sources(
        workdir / "sources.yaml", [{"slug": "tricky", "topic": "maintenance"}]
    )

    _run(workdir)

    meta = _read_frontmatter(workdir / "processed" / "tricky.md")
    assert meta["title"] == tricky


def test_failed_page_is_skipped_and_others_still_run(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    html = _page("<p>body</p>")

    def fake_fetch(url: str) -> str:
        if "missing-page" in url:
            request = httpx.Request("GET", url)
            response = httpx.Response(404, request=request)
            raise httpx.HTTPStatusError("not found", request=request, response=response)
        return html

    monkeypatch.setattr(fpd, "fetch_html", fake_fetch)
    _write_sources(
        workdir / "sources.yaml",
        [
            {"slug": "missing-page", "topic": "maintenance"},
            {"slug": "present-page", "topic": "monitoring"},
        ],
    )

    _run(workdir)

    processed = workdir / "processed"
    assert (processed / "present-page.md").exists()
    assert not (processed / "missing-page.md").exists()
    assert not (workdir / "raw" / "missing-page.html").exists()
