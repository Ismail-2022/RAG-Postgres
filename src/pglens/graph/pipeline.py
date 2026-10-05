"""Question answering over the PostgreSQL docs: retrieve, generate, then check citations.

    retrieve -> generate -> cite_check

- retrieve: hybrid search over the indexed chunks.
- generate: the LLM answers from the numbered sources and cites them as [n].
- cite_check: every citation must point at a source that was actually given.
  Anything else is refused, so a made-up citation never stands as an answer.

`stream_answer` runs the same steps, but yields the model's text as it is
written. The text is only final once `cite_check` passes, so the caller must
treat streamed text as a draft until the final event arrives.

A reranking step will go between retrieve and generate once a reranker is chosen.

    python -m pglens.graph.pipeline "how do I find blocked queries?"
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from qdrant_client import QdrantClient

from pglens.config import INDEX_DIR
from pglens.llm import LLM
from pglens.retrieval.indexer import COLLECTION, Embedder, search

TOP_K = 5
REFUSAL = "I can't answer this from the PostgreSQL documentation."
CITATION = re.compile(r"\[(\d+)\]")

SYSTEM_RULES = (
    "You answer questions about PostgreSQL using only the numbered sources below.\n"
    "Every sentence must end with a citation to the source it comes from, like [1] "
    "or [2][3]. Do not write a sentence that has no citation.\n"
    "Do not write lead-in lines that end in a colon.\n"
    f"If the sources do not answer the question, reply exactly: {REFUSAL}\n"
    "Do not use knowledge from outside the sources."
)


class PipelineState(TypedDict, total=False):
    question: str
    hits: list[dict[str, Any]]
    answer: str
    sources: list[dict[str, Any]]
    refused: bool
    reason: str


@dataclass(frozen=True)
class Answer:
    question: str
    answer: str
    sources: tuple[dict[str, Any], ...]
    refused: bool
    reason: str
    hits: tuple[
        dict[str, Any], ...
    ] = ()  # everything retrieved, so a judge can read the sources


def build_prompt(question: str, hits: list[dict[str, Any]]) -> str:
    blocks = []
    for number, hit in enumerate(hits, start=1):
        heading = " > ".join(hit["heading_path"]) or hit["title"]
        blocks.append(f"[{number}] {hit['title']} — {heading}\n{hit['text']}")
    sources = "\n\n".join(blocks)
    return f"{SYSTEM_RULES}\n\nSources:\n\n{sources}\n\nQuestion: {question}\nAnswer:"


def check_citations(answer: str, hits: list[dict[str, Any]]) -> tuple[bool, str]:
    """Return (ok, reason). An answer passes only if it cites at least one source,
    and every citation number refers to a source that was given."""
    if answer == REFUSAL:
        return False, "model declined: sources do not answer the question"
    cited = {int(n) for n in CITATION.findall(answer)}
    if not cited:
        return False, "answer has no citations"
    unknown = sorted(n for n in cited if not 1 <= n <= len(hits))
    if unknown:
        return False, f"answer cites sources that do not exist: {unknown}"
    return True, ""


def retrieve_hits(
    client: QdrantClient, embedder: Embedder, question: str, top_k: int = TOP_K
) -> list[dict[str, Any]]:
    if not client.collection_exists(COLLECTION):
        raise RuntimeError(
            "the vector index has not been built; run "
            "`python -m pglens.retrieval.indexer` first"
        )
    return search(client, embedder, question, limit=top_k, mode="hybrid")


def finalise(question: str, hits: list[dict[str, Any]], text: str) -> Answer:
    """Turn the model's raw text into the answer the user sees, or a refusal."""
    if not hits:
        return Answer(
            question, REFUSAL, (), True, "no matching documentation was found", ()
        )
    text = text.strip()
    ok, reason = check_citations(text, hits)
    if not ok:
        return Answer(question, REFUSAL, (), True, reason, tuple(hits))
    cited = sorted({int(n) for n in CITATION.findall(text)})
    sources = tuple(
        {
            "number": n,
            "title": hits[n - 1]["title"],
            "heading_path": hits[n - 1]["heading_path"],
            "source_url": hits[n - 1]["source_url"],
        }
        for n in cited
    )
    return Answer(question, text, sources, False, "", tuple(hits))


def build_graph(
    client: QdrantClient, embedder: Embedder, llm: LLM, top_k: int = TOP_K
) -> Any:
    def retrieve(state: PipelineState) -> PipelineState:
        return {"hits": retrieve_hits(client, embedder, state["question"], top_k)}

    def generate(state: PipelineState) -> PipelineState:
        if not state["hits"]:
            return {"answer": ""}
        prompt = build_prompt(state["question"], state["hits"])
        return {"answer": llm.complete(prompt)}

    def cite_check(state: PipelineState) -> PipelineState:
        result = finalise(state["question"], state["hits"], state.get("answer", ""))
        return {
            "answer": result.answer,
            "sources": [dict(s) for s in result.sources],
            "refused": result.refused,
            "reason": result.reason,
        }

    graph = StateGraph(PipelineState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("cite_check", cite_check)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "cite_check")
    graph.add_edge("cite_check", END)
    return graph.compile()


def ask(
    question: str,
    client: QdrantClient,
    embedder: Embedder,
    llm: LLM,
    top_k: int = TOP_K,
) -> Answer:
    graph = build_graph(client, embedder, llm, top_k)
    state = graph.invoke({"question": question})
    return Answer(
        question=question,
        answer=state["answer"],
        sources=tuple(state.get("sources", [])),
        refused=bool(state.get("refused")),
        reason=state.get("reason", ""),
        hits=tuple(state.get("hits", [])),
    )


def stream_answer(
    question: str,
    client: QdrantClient,
    embedder: Embedder,
    llm: LLM,
    top_k: int = TOP_K,
) -> Iterator[dict[str, Any]]:
    """Yield events as the answer is produced.

    retrieved -> token (many) -> final

    "token" text is a draft. The "final" event carries the verified answer,
    which may differ from the draft if its citations fail.
    """
    hits = retrieve_hits(client, embedder, question, top_k)
    yield {"type": "retrieved", "count": len(hits)}

    if not hits:
        yield {"type": "final", "answer": finalise(question, hits, "")}
        return

    pieces: list[str] = []
    for piece in llm.stream(build_prompt(question, hits)):
        pieces.append(piece)
        yield {"type": "token", "text": piece}

    yield {"type": "final", "answer": finalise(question, hits, "".join(pieces))}


def format_answer(result: Answer) -> str:
    lines = [result.answer]
    if result.sources:
        lines.append("")
        lines.append("Sources:")
        for source in result.sources:
            heading = " > ".join(source["heading_path"]) or source["title"]
            lines.append(f"  [{source['number']}] {heading}  {source['source_url']}")
    if result.refused:
        lines.append("")
        lines.append(f"(refused: {result.reason})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    from pglens.llm.ollama import OllamaLLM
    from pglens.retrieval import FastEmbedder, open_index

    args = sys.argv[1:] if argv is None else argv
    if not args:
        raise SystemExit('usage: python -m pglens.graph.pipeline "your question"')
    result = ask(args[0], open_index(INDEX_DIR), FastEmbedder(), OllamaLLM())
    print(format_answer(result))


if __name__ == "__main__":
    main()
