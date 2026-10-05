"""Tests for the web API, using an in-memory index and a scripted model."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from support import FakeEmbedder

from pglens.api import create_app
from pglens.chunking import Chunk
from pglens.graph.pipeline import REFUSAL
from pglens.retrieval.indexer import index_chunks


class ScriptedLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply

    def complete(self, prompt: str) -> str:
        return self.reply

    def stream(self, prompt: str):
        yield self.reply


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
        heading_path=("27.3. Viewing Locks",),
        text="pg_locks shows which sessions hold or wait for locks",
    )
    index_chunks(client, [chunk], FakeEmbedder())
    return client


def _client(index: QdrantClient, reply: str) -> TestClient:
    app = create_app(index, FakeEmbedder(), ScriptedLLM(reply))
    return TestClient(app)


def test_health_reports_ok(index: QdrantClient) -> None:
    response = _client(index, "unused").get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_returns_a_cited_answer_with_its_sources(index: QdrantClient) -> None:
    client = _client(index, "Query pg_locks to see waiting sessions [1].")

    response = client.post("/ask", json={"question": "how do I see blocked sessions"})
    body = response.json()

    assert response.status_code == 200
    assert body["refused"] is False
    assert body["answer"] == "Query pg_locks to see waiting sessions [1]."
    assert body["sources"] == [
        {
            "number": 1,
            "title": "Viewing Locks",
            "heading_path": ["27.3. Viewing Locks"],
            "source_url": "https://www.postgresql.org/docs/17/monitoring-locks.html",
        }
    ]


def test_ask_returns_a_refusal_as_a_normal_response(index: QdrantClient) -> None:
    client = _client(index, REFUSAL)

    body = client.post(
        "/ask", json={"question": "what is the capital of France"}
    ).json()

    assert body["refused"] is True
    assert body["answer"] == REFUSAL
    assert body["sources"] == []


def test_question_that_is_too_short_is_rejected(index: QdrantClient) -> None:
    response = _client(index, "unused").post("/ask", json={"question": "hi"})

    assert response.status_code == 422


def test_question_that_is_too_long_is_rejected(index: QdrantClient) -> None:
    response = _client(index, "unused").post("/ask", json={"question": "x" * 501})

    assert response.status_code == 422


def test_missing_index_returns_503_with_the_fix(index: QdrantClient) -> None:
    client = _client(QdrantClient(":memory:"), "unused")

    response = client.post("/ask", json={"question": "how do I see blocked sessions"})

    assert response.status_code == 503
    assert "indexer" in response.json()["detail"]


def test_home_page_is_served(index: QdrantClient) -> None:
    response = _client(index, "unused").get("/")

    assert response.status_code == 200
    assert "PGLens" in response.text
    assert 'id="composer"' in response.text
    assert 'id="question"' in response.text


def test_stream_sends_retrieved_tokens_then_final(index: QdrantClient) -> None:
    client = _client(index, "Query pg_locks to see waiting sessions [1].")

    with client.stream(
        "POST", "/ask/stream", json={"question": "blocked sessions"}
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "event: retrieved" in body
    assert "event: token" in body
    assert (
        body.index("event: retrieved")
        < body.index("event: token")
        < body.index("event: final")
    )
    final_block = body.split("event: final\ndata: ")[1].split("\n\n")[0]
    assert '"refused": false' in final_block
    assert "[1]" in final_block


def test_stream_refuses_when_the_model_declines(index: QdrantClient) -> None:
    client = _client(index, REFUSAL)

    with client.stream(
        "POST", "/ask/stream", json={"question": "capital of France"}
    ) as response:
        body = "".join(response.iter_text())

    assert '"refused": true' in body


def test_stream_returns_503_when_index_is_missing() -> None:
    client = _client(QdrantClient(":memory:"), "unused")

    response = client.post(
        "/ask/stream", json={"question": "how do I see blocked sessions"}
    )

    assert response.status_code == 503
