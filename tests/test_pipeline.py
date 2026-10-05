"""Tests for the question-answering pipeline.

A scripted fake LLM stands in for Ollama, so these run offline. The rules
under test are the ones that keep wrong answers from reaching the user:
citations must point at real sources, and anything else is refused.
"""

from __future__ import annotations

import httpx
import pytest
from qdrant_client import QdrantClient
from support import FakeEmbedder

from pglens.chunking import Chunk
from pglens.graph.pipeline import REFUSAL, ask, build_prompt, check_citations
from pglens.llm.ollama import OllamaLLM
from pglens.retrieval.indexer import index_chunks


class ScriptedLLM:
    """Returns canned replies in order and records each prompt it was sent."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.replies.pop(0)

    def stream(self, prompt: str):
        yield self.complete(prompt)


def _chunk(chunk_id: str, text: str, slug: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        slug=slug,
        title=f"Title of {slug}",
        topic="test-topic",
        source_url=f"https://www.postgresql.org/docs/17/{slug}.html",
        docs_version="17",
        heading_path=("Section",),
        text=text,
    )


@pytest.fixture
def index() -> QdrantClient:
    client = QdrantClient(":memory:")
    index_chunks(
        client,
        [
            _chunk(
                "a#0",
                "pg_locks shows which sessions hold or wait for locks",
                "monitoring-locks",
            ),
            _chunk("b#0", "a btree index speeds up equality lookups", "indexes-types"),
        ],
        FakeEmbedder(),
    )
    return client


# --- citation rules ---------------------------------------------------------


def test_answer_with_valid_citation_passes() -> None:
    hits = [{"text": "x"}, {"text": "y"}]

    assert check_citations("Use pg_locks [1].", hits) == (True, "")


def test_answer_without_citations_is_rejected() -> None:
    ok, reason = check_citations("Use pg_locks.", [{"text": "x"}])

    assert not ok
    assert "no citations" in reason


def test_citation_to_a_missing_source_is_rejected() -> None:
    ok, reason = check_citations("Use pg_locks [1][7].", [{"text": "x"}, {"text": "y"}])

    assert not ok
    assert "[7]" in reason


def test_citation_zero_is_rejected() -> None:
    ok, _ = check_citations("Use pg_locks [0].", [{"text": "x"}])

    assert not ok


def test_models_refusal_is_recognised() -> None:
    ok, reason = check_citations(REFUSAL, [{"text": "x"}])

    assert not ok
    assert "declined" in reason


def test_prompt_numbers_every_source_and_includes_the_question() -> None:
    hits = [
        {"title": "T1", "heading_path": ["A", "B"], "text": "first text"},
        {"title": "T2", "heading_path": [], "text": "second text"},
    ]

    prompt = build_prompt("my question", hits)

    assert "[1] T1 — A > B\nfirst text" in prompt
    assert "[2] T2 — T2\nsecond text" in prompt
    assert "Question: my question" in prompt


# --- the pipeline end to end, offline ----------------------------------------


def test_cited_answer_returns_sources_with_urls(index: QdrantClient) -> None:
    llm = ScriptedLLM("Query pg_locks to see waiting sessions [1].")

    result = ask("how do I see blocked sessions", index, FakeEmbedder(), llm)

    assert not result.refused
    assert result.answer == "Query pg_locks to see waiting sessions [1]."
    assert [s["slug"] if "slug" in s else s["source_url"] for s in result.sources] == [
        "https://www.postgresql.org/docs/17/monitoring-locks.html"
    ]


def test_only_cited_sources_are_listed(index: QdrantClient) -> None:
    llm = ScriptedLLM("Answer from the second source only [2].")

    result = ask("btree index lookups", index, FakeEmbedder(), llm, top_k=2)

    assert [s["number"] for s in result.sources] == [2]


def test_uncited_answer_is_refused_and_not_shown(index: QdrantClient) -> None:
    llm = ScriptedLLM("Trust me, it is pg_locks.")

    result = ask("blocked sessions", index, FakeEmbedder(), llm)

    assert result.refused
    assert result.answer == REFUSAL
    assert result.sources == ()


def test_model_refusal_passes_through_as_refused(index: QdrantClient) -> None:
    llm = ScriptedLLM(REFUSAL)

    result = ask("what is the weather", index, FakeEmbedder(), llm)

    assert result.refused
    assert result.answer == REFUSAL


def test_missing_index_gives_a_clear_error_before_any_model_call() -> None:
    llm = ScriptedLLM()  # any call would pop from an empty list and fail

    with pytest.raises(RuntimeError, match="indexer"):
        ask("anything", QdrantClient(":memory:"), FakeEmbedder(), llm)

    assert llm.prompts == []


def test_prompt_sent_to_the_model_contains_the_retrieved_text(
    index: QdrantClient,
) -> None:
    llm = ScriptedLLM("Use pg_locks [1].")

    ask("blocked sessions", index, FakeEmbedder(), llm)

    assert len(llm.prompts) == 1
    assert "pg_locks shows which sessions" in llm.prompts[0]


# --- Ollama client, with the HTTP layer mocked -------------------------------


def test_ollama_client_sends_the_expected_request_and_returns_text() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = request.read()
        return httpx.Response(200, json={"response": "  hello [1]  "})

    llm = OllamaLLM(model="test-model", transport=httpx.MockTransport(handler))

    assert llm.complete("prompt text") == "hello [1]"
    assert seen["path"] == "/api/generate"
    body = str(seen["body"])
    assert '"model":"test-model"' in body
    assert '"stream":false' in body
    assert '"num_ctx":8192' in body


def test_ollama_http_errors_are_raised() -> None:
    llm = OllamaLLM(transport=httpx.MockTransport(lambda r: httpx.Response(500)))

    with pytest.raises(httpx.HTTPStatusError):
        llm.complete("prompt")
