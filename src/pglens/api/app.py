"""Web API and page for asking questions about the PostgreSQL docs.

    python -m pglens.api        # then open http://127.0.0.1:8000

Endpoints:

- GET  /        the question page
- POST /ask     {"question": "..."} -> answer, refusal flag, and cited sources
- GET  /health  liveness check
"""

from __future__ import annotations

import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from pglens.graph.pipeline import Answer, ask
from pglens.llm import LLM
from pglens.retrieval.indexer import Embedder

STATIC_DIR = Path(__file__).parent / "static"
MAX_QUESTION_CHARS = 500


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=MAX_QUESTION_CHARS)


class Source(BaseModel):
    number: int
    title: str
    heading_path: list[str]
    source_url: str


class AskResponse(BaseModel):
    question: str
    answer: str
    refused: bool
    reason: str
    sources: list[Source]


def to_response(result: Answer) -> AskResponse:
    return AskResponse(
        question=result.question,
        answer=result.answer,
        refused=result.refused,
        reason=result.reason,
        sources=[Source(**source) for source in result.sources],
    )


def create_app(client: QdrantClient, embedder: Embedder, llm: LLM) -> FastAPI:
    app = FastAPI(title="PGLens", version="0.1.0")
    # The embedded Qdrant client is not safe to share between threads
    lock = threading.Lock()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ask", response_model=AskResponse)
    def ask_endpoint(body: AskRequest) -> AskResponse:
        try:
            with lock:
                result = ask(body.question, client, embedder, llm)
        except RuntimeError as error:  # raised when the index has not been built
            raise HTTPException(status_code=503, detail=str(error)) from error
        return to_response(result)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


def default_app() -> FastAPI:
    from pglens.config import INDEX_DIR
    from pglens.llm.ollama import OllamaLLM
    from pglens.retrieval import FastEmbedder, open_index

    return create_app(open_index(INDEX_DIR), FastEmbedder(), OllamaLLM())


def main() -> None:
    import uvicorn

    uvicorn.run(default_app(), host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
