"""Split processed docs markdown into heading-aware chunks for indexing.

Each chunk is one section of a page, or one piece of a section that is too
long. Every chunk keeps the page metadata and its heading path, so an answer
can cite the exact section it came from.

Code blocks and tables are never split, even when they exceed the size limit,
because a cut in the middle would change their meaning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

MAX_CHARS = 3500  # roughly 800 tokens of English prose

FENCE = "```"
HEADING = re.compile(r"^(#{1,6}) +(.+?) *$")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    slug: str
    title: str
    topic: str
    source_url: str
    docs_version: str
    heading_path: tuple[str, ...]
    text: str


def split_frontmatter(document: str) -> tuple[dict[str, Any], str]:
    """Return (frontmatter, body) for a processed markdown document."""
    if not document.startswith("---\n"):
        raise ValueError("document has no YAML frontmatter")
    _, frontmatter, body = document.split("---\n", 2)
    meta: dict[str, Any] = yaml.safe_load(frontmatter)
    return meta, body.lstrip("\n")


def chunk_document(document: str, max_chars: int = MAX_CHARS) -> list[Chunk]:
    """Split one processed markdown document into chunks."""
    meta, body = split_frontmatter(document)
    chunks: list[Chunk] = []
    for path, heading, lines in _sections(body):
        # Every piece of a section repeats its heading, so a chunk makes sense alone
        prefix = f"{heading}\n\n" if heading else ""
        room = max_chars - len(prefix)
        for piece in _pack(_blocks(lines), room):
            chunks.append(
                Chunk(
                    chunk_id=f"{meta['slug']}#{len(chunks)}",
                    slug=str(meta["slug"]),
                    title=str(meta["title"]),
                    topic=str(meta.get("topic", "")),
                    source_url=str(meta["source_url"]),
                    docs_version=str(meta["docs_version"]),
                    heading_path=path,
                    text=prefix + piece,
                )
            )
    return chunks


def chunk_processed_dir(directory: Path, max_chars: int = MAX_CHARS) -> list[Chunk]:
    """Chunk every markdown file in a directory, in filename order."""
    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*.md")):
        chunks.extend(chunk_document(path.read_text(), max_chars))
    return chunks


def _sections(body: str) -> list[tuple[tuple[str, ...], str, list[str]]]:
    """Split the body at headings. Returns (heading_path, heading_line, body_lines).

    Headings inside code fences are ignored. Sections with no text under
    their heading are dropped.
    """
    sections: list[tuple[tuple[str, ...], str, list[str]]] = []
    stack: list[tuple[int, str]] = []
    path: tuple[str, ...] = ()
    heading = ""
    lines: list[str] = []
    in_fence = False

    for line in body.splitlines():
        if line.startswith(FENCE):
            in_fence = not in_fence
        elif not in_fence and (match := HEADING.match(line)):
            sections.append((path, heading, lines))
            level = len(match.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, match.group(2)))
            path = tuple(text for _, text in stack)
            heading = line
            lines = []
            continue
        lines.append(line)
    sections.append((path, heading, lines))

    return [s for s in sections if any(line.strip() for line in s[2])]


def _blocks(lines: list[str]) -> list[str]:
    """Group lines into blocks that must stay together.

    A block is a paragraph, a whole code fence, or a run of table rows.
    Blocks are separated by blank lines.
    """
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False

    def close() -> None:
        if current:
            blocks.append("\n".join(current).strip("\n"))
            current.clear()

    for line in lines:
        if line.startswith(FENCE):
            if not in_fence:
                close()
            in_fence = not in_fence
            current.append(line)
            if not in_fence:
                close()
            continue
        if in_fence:
            current.append(line)
            continue
        if not line.strip():
            close()
            continue
        is_table = line.startswith("|")
        if current and is_table != current[-1].startswith("|"):
            close()
        current.append(line)
    close()

    return [block for block in blocks if block]


def _pack(blocks: list[str], room: int) -> list[str]:
    """Join blocks into pieces no longer than room.

    A single block longer than room becomes a piece on its own.
    """
    pieces: list[str] = []
    current = ""
    for block in blocks:
        candidate = f"{current}\n\n{block}" if current else block
        if current and len(candidate) > room:
            pieces.append(current)
            current = block
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces
