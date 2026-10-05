"""Score answer quality with a local LLM judge.

Two checks, both run through the same pipeline a user would hit:

- Refusal accuracy: questions the docs cannot answer should be refused, and
  answerable questions should not be.
- Faithfulness: each claim in an answer must be supported by the source(s)
  it cites. Uncited claims count as unsupported.

The judge is the local model, so it can share the generator's blind spots.
Treat the scores as a trend to watch, not an absolute measure.

Run from the project root:

    python -m pglens.eval.answer_eval
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from pglens.config import EVAL_DIR, INDEX_DIR
from pglens.graph.pipeline import CITATION, Answer, ask
from pglens.llm import LLM
from pglens.retrieval.indexer import Embedder

ANSWER_SET_FILE = EVAL_DIR / "answer_set.jsonl"

JUDGE_RULES = (
    "You check whether a claim is supported by the source text.\n"
    "Reply with exactly one word: SUPPORTED if the source states or directly "
    "implies the claim, otherwise NOT_SUPPORTED."
)


@dataclass(frozen=True)
class AnswerItem:
    id: str
    question: str
    expect_refusal: bool


@dataclass(frozen=True)
class ClaimVerdict:
    claim: str
    citations: tuple[int, ...]
    supported: bool

    @property
    def cited(self) -> bool:
        return bool(self.citations)


@dataclass(frozen=True)
class QuestionScore:
    item: AnswerItem
    answer: Answer
    refused: bool
    refusal_correct: bool
    verdicts: tuple[ClaimVerdict, ...]


@dataclass(frozen=True)
class AnswerReport:
    scores: tuple[QuestionScore, ...]
    refusal_accuracy: float
    faithfulness: float | None  # over cited claims; None when there are none
    citation_coverage: float | None  # share of all claims that carry a citation


def load_answer_set(path: Path = ANSWER_SET_FILE) -> list[AnswerItem]:
    items = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        items.append(
            AnswerItem(
                id=row["id"],
                question=row["question"],
                expect_refusal=row["expect"] == "refuse",
            )
        )
    return items


def split_claims(answer: str) -> list[str]:
    """Split an answer into sentence-level claims.

    Code blocks, lead-in lines ending in a colon, and bullet markers are not
    claims, so they are dropped.
    """
    prose: list[str] = []
    in_code = False
    for line in answer.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
            continue
        if in_code or line.rstrip().endswith(":"):
            continue
        prose.append(re.sub(r"^\s*[*-]\s+", "", line))
    parts = re.split(r"(?<=[.!?])\s+|\n+", "\n".join(prose))
    return [part.strip() for part in parts if re.search(r"[A-Za-z]", part)]


def judge_prompt(claim: str, sources: list[str]) -> str:
    source_text = "\n\n".join(sources) if sources else "(no sources cited)"
    return f"{JUDGE_RULES}\n\nSource:\n{source_text}\n\nClaim: {claim}\nReply:"


def parse_verdict(reply: str) -> bool:
    """True only for a clear SUPPORTED. Anything unclear counts as unsupported."""
    match = re.match(r"\W*(NOT_SUPPORTED|SUPPORTED)\b", reply.strip().upper())
    return bool(match) and match.group(1) == "SUPPORTED"


def judge_claim(
    llm: LLM, claim: str, citations: tuple[int, ...], hits: tuple[dict[str, Any], ...]
) -> bool:
    if not citations:
        return False  # an uncited claim is unsupported by definition
    sources = [hits[n - 1]["text"] for n in citations if 1 <= n <= len(hits)]
    return parse_verdict(llm.complete(judge_prompt(claim, sources)))


def score_answer(item: AnswerItem, answer: Answer, llm: LLM) -> QuestionScore:
    refusal_correct = answer.refused == item.expect_refusal
    verdicts: tuple[ClaimVerdict, ...] = ()
    if not answer.refused:
        verdicts = tuple(
            _verdict(claim, answer.hits, llm) for claim in split_claims(answer.answer)
        )
    return QuestionScore(item, answer, answer.refused, refusal_correct, verdicts)


def _verdict(claim: str, hits: tuple[dict[str, Any], ...], llm: LLM) -> ClaimVerdict:
    citations = tuple(sorted({int(n) for n in CITATION.findall(claim)}))
    return ClaimVerdict(claim, citations, judge_claim(llm, claim, citations, hits))


def evaluate_answers(
    items: list[AnswerItem],
    client: QdrantClient,
    embedder: Embedder,
    llm: LLM,
) -> AnswerReport:
    scores = tuple(
        score_answer(item, ask(item.question, client, embedder, llm), llm)
        for item in items
    )
    return summarise(scores)


def summarise(scores: tuple[QuestionScore, ...]) -> AnswerReport:
    refusal_accuracy = (
        sum(s.refusal_correct for s in scores) / len(scores) if scores else 0.0
    )
    claims = [v for s in scores for v in s.verdicts]
    cited = [v for v in claims if v.cited]
    faithfulness = sum(v.supported for v in cited) / len(cited) if cited else None
    coverage = len(cited) / len(claims) if claims else None
    return AnswerReport(scores, refusal_accuracy, faithfulness, coverage)


def format_report(report: AnswerReport) -> str:
    lines = [
        f"{'id':6} {'expect':8} {'got':8} {'claims':>6} {'cited':>5} {'supported':>9}  question"
    ]
    for s in report.scores:
        expect = "refuse" if s.item.expect_refusal else "answer"
        got = "refused" if s.refused else "answered"
        cited = sum(v.cited for v in s.verdicts)
        supported = sum(v.supported for v in s.verdicts if v.cited)
        mark = "" if s.refusal_correct else "  <-- wrong"
        lines.append(
            f"{s.item.id:6} {expect:8} {got:8} {len(s.verdicts):>6} {cited:>5} "
            f"{supported:>9}  {s.item.question}{mark}"
        )
    lines.append("")
    lines.append(f"refusal accuracy:  {report.refusal_accuracy:.3f}")
    if report.citation_coverage is None:
        lines.append("citation coverage: n/a (no claims were made)")
    else:
        lines.append(f"citation coverage: {report.citation_coverage:.3f}")
    if report.faithfulness is None:
        lines.append("faithfulness:      n/a (no cited claims)")
    else:
        lines.append(
            f"faithfulness:      {report.faithfulness:.3f}  (cited claims only)"
        )
    unsupported = [
        (s.item.id, v.claim)
        for s in report.scores
        for v in s.verdicts
        if v.cited and not v.supported
    ]
    if unsupported:
        lines.append("")
        lines.append("cited but unsupported:")
        for qid, claim in unsupported:
            lines.append(f"  {qid}: {claim}")
    uncited = [
        (s.item.id, v.claim) for s in report.scores for v in s.verdicts if not v.cited
    ]
    if uncited:
        lines.append("")
        lines.append("uncited claims:")
        for qid, claim in uncited:
            lines.append(f"  {qid}: {claim}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    from pglens.llm.ollama import OllamaLLM
    from pglens.retrieval import FastEmbedder, open_index

    args = sys.argv[1:] if argv is None else argv
    path = Path(args[0]) if args else ANSWER_SET_FILE
    items = load_answer_set(path)
    llm = OllamaLLM()
    print(f"answer set: {path.name} ({len(items)} questions), model: {llm.model}")
    report = evaluate_answers(items, open_index(INDEX_DIR), FastEmbedder(), llm)
    print(format_report(report))


if __name__ == "__main__":
    main()
