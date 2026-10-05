"""Tests for the retrieval scoring.

Metric tests use synthetic hits, so they check the arithmetic and matching
rules in isolation. The golden-set test checks the hand-written questions
against the real corpus, so a mistyped slug or heading fails here instead of
quietly scoring zero.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path
from typing import Any

import pytest
import yaml
from qdrant_client import QdrantClient

from pglens.chunking import Chunk, chunk_processed_dir
from pglens.config import EVAL_DIR, PROCESSED_DIR, SOURCES_FILE
from pglens.eval.retrieval_eval import (
    GoldenQuestion,
    Target,
    evaluate,
    first_relevant_rank,
    is_relevant,
    load_golden_set,
    mean_reciprocal_rank,
    recall_at_k,
)
from pglens.retrieval.indexer import index_chunks

DIMENSIONS = 64


class FakeEmbedder:
    """Deterministic bag-of-words embedding, as in the indexer tests."""

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * DIMENSIONS
        for word in text.lower().split():
            vector[zlib.crc32(word.encode()) % DIMENSIONS] += 1.0
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def _hit(slug: str, path: tuple[str, ...] = ()) -> dict[str, Any]:
    return {"slug": slug, "heading_path": list(path)}


# --- matching rules ---------------------------------------------------------


def test_slug_must_match() -> None:
    assert not is_relevant(_hit("other"), Target(slug="monitoring-locks"))


def test_page_target_matches_any_section_of_that_page() -> None:
    assert is_relevant(
        _hit("explicit-joins", ("Anything",)), Target(slug="explicit-joins")
    )


def test_heading_target_matches_within_the_heading_path() -> None:
    hit = _hit("routine-vacuuming", ("Routine Vacuuming", "The Autovacuum Daemon"))

    assert is_relevant(hit, Target(slug="routine-vacuuming", heading="Autovacuum"))


def test_heading_match_is_case_insensitive() -> None:
    hit = _hit("routine-vacuuming", ("The Autovacuum Daemon",))

    assert is_relevant(
        hit, Target(slug="routine-vacuuming", heading="autovacuum daemon")
    )


def test_heading_target_rejects_other_sections_of_the_same_page() -> None:
    hit = _hit("routine-vacuuming", ("Routine Vacuuming", "Recovering Disk Space"))

    assert not is_relevant(hit, Target(slug="routine-vacuuming", heading="Autovacuum"))


# --- ranks and metrics ------------------------------------------------------


def test_first_relevant_rank_is_one_based() -> None:
    question = GoldenQuestion("q", "?", (Target(slug="b"),))
    hits = [_hit("a"), _hit("b"), _hit("b")]

    assert first_relevant_rank(hits, question) == 2


def test_first_relevant_rank_is_none_when_missed() -> None:
    question = GoldenQuestion("q", "?", (Target(slug="b"),))

    assert first_relevant_rank([_hit("a"), _hit("c")], question) is None


def test_any_of_several_targets_counts_as_a_hit() -> None:
    question = GoldenQuestion("q", "?", (Target(slug="x"), Target(slug="y")))

    assert first_relevant_rank([_hit("z"), _hit("y")], question) == 2


def test_recall_at_k_counts_ranks_within_k() -> None:
    ranks = [1, 3, 6, None]

    assert recall_at_k(ranks, k=3) == 0.5
    assert recall_at_k(ranks, k=5) == 0.5
    assert recall_at_k(ranks, k=6) == 0.75


def test_mrr_averages_reciprocal_ranks_and_counts_misses_as_zero() -> None:
    ranks = [1, 2, None]

    assert mean_reciprocal_rank(ranks) == pytest.approx((1 + 0.5 + 0) / 3)


def test_metrics_are_zero_for_no_questions() -> None:
    assert recall_at_k([], k=5) == 0.0
    assert mean_reciprocal_rank([]) == 0.0


# --- loading and end to end -------------------------------------------------


def test_golden_set_loads_targets_with_and_without_headings(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    rows = [
        {"id": "q1", "question": "one", "relevant": [{"slug": "a", "heading": "H"}]},
        {"id": "q2", "question": "two", "relevant": [{"slug": "b"}]},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    questions = load_golden_set(path)

    assert questions[0].relevant == (Target(slug="a", heading="H"),)
    assert questions[1].relevant == (Target(slug="b", heading=None),)


def test_evaluate_scores_a_perfect_and_a_missed_question() -> None:
    client = QdrantClient(":memory:")
    chunks = [
        Chunk(
            "vac#0",
            "vac",
            "V",
            "t",
            "u",
            "17",
            ("Autovacuum",),
            "autovacuum daemon runs",
        ),
        Chunk("idx#0", "idx", "I", "t", "u", "17", ("Btree",), "btree index lookups"),
    ]
    index_chunks(client, chunks, FakeEmbedder())
    questions = [
        GoldenQuestion(
            "hit", "when does autovacuum run", (Target("vac", "Autovacuum"),)
        ),
        GoldenQuestion("miss", "unrelated words", (Target("nonexistent"),)),
    ]

    report = evaluate(client, FakeEmbedder(), questions, k=1)

    assert report.results[0].rank == 1
    assert report.results[1].rank is None
    assert report.recall_at_k == 0.5
    assert report.mrr == pytest.approx(0.5)


# --- the golden set against the real corpus ---------------------------------


def _configured_slugs() -> set[str]:
    config = yaml.safe_load(SOURCES_FILE.read_text())
    return {page["slug"] for page in config["pages"]}


GOLDEN_SETS = sorted(EVAL_DIR.glob("golden_set*.jsonl"))


@pytest.mark.parametrize("golden", GOLDEN_SETS, ids=lambda p: p.stem)
def test_golden_set_ids_are_unique(golden: Path) -> None:
    ids = [q.id for q in load_golden_set(golden)]

    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("golden", GOLDEN_SETS, ids=lambda p: p.stem)
def test_golden_set_targets_exist_in_the_corpus(golden: Path) -> None:
    """Every slug is configured, and every heading appears on that page.

    A typo or a renamed heading fails here, rather than scoring as a miss.
    """
    if not PROCESSED_DIR.exists():
        pytest.skip("data/processed not present; run the ingestion first")

    configured = _configured_slugs()
    headings: dict[str, list[str]] = {}
    for chunk in chunk_processed_dir(PROCESSED_DIR):
        headings.setdefault(chunk.slug, []).append(
            " > ".join(chunk.heading_path).lower()
        )

    problems = []
    for question in load_golden_set(golden):
        for target in question.relevant:
            if target.slug not in configured:
                problems.append(
                    f"{question.id}: slug {target.slug!r} not in sources.yaml"
                )
                continue
            if target.heading is None:
                continue
            needle = target.heading.lower()
            if not any(needle in path for path in headings.get(target.slug, [])):
                problems.append(
                    f"{question.id}: heading {target.heading!r} not on {target.slug}"
                )

    assert problems == []
