"""Score retrieval against the golden set.

A question is answered by a retrieved chunk when the chunk comes from the
right page and, if the target names a heading, that heading appears in the
chunk's heading path. Scores:

- recall@k: share of questions with a relevant chunk in the top k
- MRR: mean reciprocal rank of the first relevant chunk (0 if none in the top N)

Run from the project root:

    python -m pglens.eval.retrieval_eval
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from pglens.config import GOLDEN_SET_FILE, INDEX_DIR
from pglens.retrieval.indexer import MODES, Embedder, Mode, search

SEARCH_DEPTH = 10  # how many results to retrieve per question
RECALL_K = 5  # the k used for the headline recall score


@dataclass(frozen=True)
class Target:
    slug: str
    heading: str | None = None


@dataclass(frozen=True)
class GoldenQuestion:
    id: str
    question: str
    relevant: tuple[Target, ...]


@dataclass(frozen=True)
class QuestionResult:
    question: GoldenQuestion
    rank: int | None  # 1-based rank of the first relevant hit, None if missed


@dataclass(frozen=True)
class EvalReport:
    results: tuple[QuestionResult, ...]
    recall_at_k: float
    mrr: float
    k: int


def load_golden_set(path: Path = GOLDEN_SET_FILE) -> list[GoldenQuestion]:
    questions = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        targets = tuple(Target(**t) for t in row["relevant"])
        questions.append(GoldenQuestion(row["id"], row["question"], targets))
    return questions


def is_relevant(hit: dict[str, Any], target: Target) -> bool:
    if hit["slug"] != target.slug:
        return False
    if target.heading is None:
        return True
    return target.heading.lower() in " > ".join(hit["heading_path"]).lower()


def first_relevant_rank(
    hits: list[dict[str, Any]], question: GoldenQuestion
) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if any(is_relevant(hit, target) for target in question.relevant):
            return rank
    return None


def recall_at_k(ranks: list[int | None], k: int) -> float:
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mean_reciprocal_rank(ranks: list[int | None]) -> float:
    if not ranks:
        return 0.0
    return sum(1 / r for r in ranks if r is not None) / len(ranks)


def evaluate(
    client: QdrantClient,
    embedder: Embedder,
    questions: list[GoldenQuestion],
    k: int = RECALL_K,
    mode: Mode = "hybrid",
) -> EvalReport:
    results = []
    for question in questions:
        hits = search(
            client, embedder, question.question, limit=SEARCH_DEPTH, mode=mode
        )
        results.append(QuestionResult(question, first_relevant_rank(hits, question)))

    ranks = [r.rank for r in results]
    return EvalReport(
        results=tuple(results),
        recall_at_k=recall_at_k(ranks, k),
        mrr=mean_reciprocal_rank(ranks),
        k=k,
    )


def format_report(report: EvalReport) -> str:
    lines = [f"{'id':5} {'rank':>4}  question"]
    for result in report.results:
        rank = str(result.rank) if result.rank is not None else "miss"
        lines.append(f"{result.question.id:5} {rank:>4}  {result.question.question}")
    lines.append("")
    lines.append(f"recall@{report.k}: {report.recall_at_k:.3f}")
    lines.append(f"MRR (top {SEARCH_DEPTH}): {report.mrr:.3f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    from pglens.retrieval import FastEmbedder, open_index

    args = sys.argv[1:] if argv is None else argv
    path = Path(args[0]) if args else GOLDEN_SET_FILE
    mode = args[1] if len(args) > 1 else "hybrid"
    if mode not in MODES:
        raise SystemExit(f"mode must be one of {', '.join(MODES)}")
    questions = load_golden_set(path)
    print(f"golden set: {path.name} ({len(questions)} questions), mode: {mode}")
    report = evaluate(open_index(INDEX_DIR), FastEmbedder(), questions, mode=mode)
    print(format_report(report))


if __name__ == "__main__":
    main()
