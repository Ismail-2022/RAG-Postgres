"""Fetch curated PostgreSQL doc pages, cache raw HTML, convert to clean markdown.

    uv add httpx beautifulsoup4 pyyaml markdownify tenacity
    python fetch_postgres_docs.py

Raw HTML -> data/raw/<slug>.html   (immutable, DVC-tracked, your ground truth)
Clean MD -> data/processed/<slug>.md (frontmatter + body, this is what gets chunked)
"""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import httpx
import yaml
from bs4 import BeautifulSoup
from markdownify import markdownify
from tenacity import retry, stop_after_attempt, wait_exponential

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
MIN_INTERVAL = 0.5  # be polite, this is a shared community resource

client = httpx.Client(
    timeout=30,
    follow_redirects=True,
    headers={"User-Agent": "pglens-portfolio-project (github.com/<you>/pglens)"},
)
_last_call = 0.0


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, max=20))
def fetch_html(url: str) -> str:
    global _last_call
    wait = MIN_INTERVAL - (time.monotonic() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.monotonic()
    r = client.get(url)
    r.raise_for_status()
    return r.text


def extract_main(html: str, page_url: str) -> tuple[str, str]:
    """Return (title, main_content_html), stripping nav chrome and heading permalinks.

    Every href is resolved against page_url so links still work once the
    markdown is detached from the docs site (in-page #anchors become links
    to the section on postgresql.org).
    """
    soup = BeautifulSoup(html, "html.parser")
    for permalink in soup.select("a.id_link"):
        permalink.decompose()
    title = soup.find("h1") or soup.find("h2")
    # str.split() also splits on the &nbsp; the docs put after section numbers
    title_text = " ".join(title.get_text().split()) if title else "Untitled"
    main = soup.find(id="docContent") or soup.find("div", class_="sect1") or soup.body
    for junk in main.select("div.navheader, div.navfooter, a.navheader"):
        junk.decompose()
    for a in main.select("a[href]"):
        a["href"] = urljoin(page_url, a["href"])
    return title_text, str(main)


def to_markdown(main_html: str) -> str:
    # the docs use &nbsp; after section numbers; plain spaces keep citations greppable
    md = markdownify(main_html, heading_style="ATX").replace("\xa0", " ")
    # collapse runs of blank lines left over from stripped nav elements
    lines = md.splitlines()
    return "\n".join(
        line
        for i, line in enumerate(lines)
        if line.strip() or (i > 0 and lines[i - 1].strip())
    ).strip()


def run() -> None:
    cfg = yaml.safe_load(Path("sources.yaml").read_text())
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    fetched_at = datetime.now(timezone.utc).isoformat()

    for page in cfg["pages"]:
        slug, topic = page["slug"], page["topic"]
        url = f"{cfg['base_url']}/{slug}.html"
        try:
            html = fetch_html(url)
        except httpx.HTTPStatusError as e:
            print(f"FAILED {slug}: {e.response.status_code} — check slug is current")
            continue

        (RAW_DIR / f"{slug}.html").write_text(html)
        title, main_html = extract_main(html, url)
        body = to_markdown(main_html)
        content_hash = hashlib.sha256(body.encode()).hexdigest()[:12]

        meta = {
            "title": title,
            "slug": slug,
            "topic": topic,
            "source_url": url,
            "docs_version": cfg["version"],
            "license": cfg["license"],
            "fetched_at": fetched_at,
            "content_hash": content_hash,
        }
        # safe_dump escapes quotes and other characters a title may contain
        frontmatter = (
            "---\n"
            + yaml.safe_dump(meta, sort_keys=False, allow_unicode=True, width=4096)
            + "---\n\n"
        )
        (PROCESSED_DIR / f"{slug}.md").write_text(frontmatter + body)
        print(f"ok  {slug}  ({len(body)} chars)")


if __name__ == "__main__":
    run()
