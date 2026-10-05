"""Tests for the answer-quality judge and scoring.

A routing fake model replies with a fixed answer to generation prompts, and
with scripted verdicts to judge prompts. That checks the scoring rules
without a real model.
"""

from __future__ import annotations

from typing import Any

import pytest
from qdrant_client import QdrantClient
from support import FakeEmbedder

from pglens.chunking import Chunk
from pglens.eval.answer_eval import (
    ANSWER_SET_FILE,
    JUDGE_RULES,
    AnswerItem,
    evaluate_answers,
    judge_claim,
    load_answer_set,
    parse_verdict,
    score_answer,
    split_claims,
    summarise,
)
from pglens.graph.pipeline import REFUSAL, Answer
from pglens.retrieval.indexer import index_chunks


class RoutingLLM:
    """Answers generation prompts with one fixed reply, and judge prompts from a script."""

    def __init__(
        self, answer_reply: str, judge_replies: list[str] | None = None
    ) -> None:
        self.answer_reply = answer_reply
        self.judge_replies = list(judge_replies or [])
        self.judge_calls = 0

    def complete(self, prompt: str) -> str:
        if prompt.startswith(JUDGE_RULES):
            self.judge_calls += 1
            return self.judge_replies.pop(0)
        return self.answer_reply


def _answer(
    text: str, hits: list[dict[str, Any]] | None = None, refused: bool = False
) -> Answer:
    return Answer(
        question="q",
        answer=text,
        sources=(),
        refused=refused,
        reason="",
        hits=tuple(hits or []),
    )


def _hit(text: str) -> dict[str, Any]:
    return {"text": text, "title": "T", "heading_path": ["H"], "source_url": "u"}


# --- claims and verdicts ----------------------------------------------------


def test_claims_split_on_sentences_and_bullets() -> None:
    answer = "First claim [1]. Second claim [2].\n* A bullet claim [1]\n"

    assert split_claims(answer) == [
        "First claim [1].",
        "Second claim [2].",
        "A bullet claim [1]",
    ]


def test_lines_without_words_are_not_claims() -> None:
    assert split_claims("   \n`x = 1`\n") == ["`x = 1`"]


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("SUPPORTED", True),
        ("Supported.", True),
        ("NOT_SUPPORTED", False),
        ("not_supported", False),
        ("maybe", False),
        ("", False),
    ],
)
def test_verdict_parsing_is_strict(reply: str, expected: bool) -> None:
    assert parse_verdict(reply) is expected


def test_uncited_claim_is_unsupported_without_asking_the_judge() -> None:
    llm = RoutingLLM("unused", judge_replies=["SUPPORTED"])

    assert judge_claim(llm, "no citation here", (), ()) is False
    assert llm.judge_calls == 0


def test_judge_sees_only_the_cited_source_text() -> None:
    seen: list[str] = []

    class Spy(RoutingLLM):
        def complete(self, prompt: str) -> str:
            seen.append(prompt)
            return "SUPPORTED"

    hits = (_hit("cited text"), _hit("uncited text"))
    judge_claim(Spy("unused"), "a claim", (1,), hits)

    assert "cited text" in seen[0]
    assert "uncited text" not in seen[0]


# --- scoring ----------------------------------------------------------------


def test_faithfulness_is_the_share_of_supported_claims() -> None:
    hits = [_hit("source one"), _hit("source two")]
    answer = _answer("Claim one [1]. Claim two [2].", hits)
    llm = RoutingLLM("unused", judge_replies=["SUPPORTED", "NOT_SUPPORTED"])

    score = score_answer(AnswerItem("q", "?", expect_refusal=False), answer, llm)
    report = summarise((score,))

    assert [v.supported for v in score.verdicts] == [True, False]
    assert report.faithfulness == pytest.approx(0.5)


def test_refused_answer_makes_no_claims_and_is_judged_on_refusal_alone() -> None:
    answer = _answer(REFUSAL, refused=True)
    llm = RoutingLLM("unused")

    score = score_answer(AnswerItem("q", "?", expect_refusal=True), answer, llm)

    assert score.refusal_correct
    assert score.verdicts == ()
    assert llm.judge_calls == 0


def test_answering_an_out_of_scope_question_is_a_refusal_failure() -> None:
    answer = _answer("Paris [1].", [_hit("x")])
    llm = RoutingLLM("unused", judge_replies=["SUPPORTED"])

    score = score_answer(AnswerItem("q", "?", expect_refusal=True), answer, llm)

    assert not score.refusal_correct


def test_refusing_an_answerable_question_is_a_refusal_failure() -> None:
    answer = _answer(REFUSAL, refused=True)

    score = score_answer(
        AnswerItem("q", "?", expect_refusal=False), answer, RoutingLLM("x")
    )

    assert not score.refusal_correct


def test_faithfulness_is_none_when_nothing_was_claimed() -> None:
    score = score_answer(
        AnswerItem("q", "?", expect_refusal=True),
        _answer(REFUSAL, refused=True),
        RoutingLLM("x"),
    )

    assert summarise((score,)).faithfulness is None


# --- end to end, offline -----------------------------------------------------


@pytest.fixture
def index() -> QdrantClient:
    client = QdrantClient(":memory:")
    chunk = Chunk(
        chunk_id="a#0",
        slug="monitoring-locks",
        title="Viewing Locks",
        topic="monitoring",
        source_url="https://www.postgresql.org/docs/17/monitoring-locks.html",
        docs_version="17",
        heading_path=("Viewing Locks",),
        text="pg_locks shows which sessions hold or wait for locks",
    )
    index_chunks(client, [chunk], FakeEmbedder())
    return client


def test_evaluate_answers_scores_a_well_grounded_answer(index: QdrantClient) -> None:
    llm = RoutingLLM(
        "Query pg_locks to see waiting sessions [1].", judge_replies=["SUPPORTED"]
    )
    items = [AnswerItem("a1", "how do I see blocked sessions", expect_refusal=False)]

    report = evaluate_answers(items, index, FakeEmbedder(), llm)

    assert report.refusal_accuracy == 1.0
    assert report.faithfulness == 1.0


def test_evaluate_answers_flags_an_unsupported_claim(index: QdrantClient) -> None:
    llm = RoutingLLM("Locks are stored on disk [1].", judge_replies=["NOT_SUPPORTED"])
    items = [AnswerItem("a1", "how do I see blocked sessions", expect_refusal=False)]

    report = evaluate_answers(items, index, FakeEmbedder(), llm)

    assert report.faithfulness == 0.0


# --- the answer set itself ---------------------------------------------------


def test_answer_set_is_well_formed() -> None:
    items = load_answer_set(ANSWER_SET_FILE)

    assert len({i.id for i in items}) == len(items)
    assert any(i.expect_refusal for i in items)
    assert any(not i.expect_refusal for i in items)


def test_lead_in_lines_and_code_blocks_are_not_claims() -> None:
    answer = "The rules are as follows:\n```sql\nSELECT 1;\n```\nThe real claim [1]."

    assert split_claims(answer) == ["The real claim [1]."]


def test_uncited_claims_do_not_count_against_faithfulness() -> None:
    hits = [_hit("source")]
    answer = _answer("Cited and right [1]. Uncited sentence.", hits)
    llm = RoutingLLM("unused", judge_replies=["SUPPORTED"])

    score = score_answer(AnswerItem("q", "?", expect_refusal=False), answer, llm)
    report = summarise((score,))

    assert report.faithfulness == 1.0
    assert report.citation_coverage == pytest.approx(0.5)
    assert llm.judge_calls == 1  # the uncited sentence was never sent to the judge
