"""Evaluation use case: runs the labelled set through retrieval and the full pipeline.

Metrics
  recall_at_5          mean per-item share of gold ids in the top 5, per retriever
                       (answerable items only; retrieval is restricted to in-force ids)
  citation_precision   share of all returned citations whose chunk id is gold
  refusal_recall       share of should-refuse items that were refused
  false_refusal_rate   share of answerable items that were refused
  as_of_accuracy       share of items with an explicit as_of that were answered, cite
                       an in-force gold paragraph and cite nothing not in force
  superseded_citations count of citations (any item) not in force on the item's date
  groundedness         mean Judge score over answered items

Refusal threshold sweep: every midpoint between consecutive observed top-similarity
values is a candidate. Among candidates whose false-refusal rate stays within
``MAX_FALSE_REFUSAL`` (the operating budget), the selected threshold maximises refusal
recall; ties are broken by the widest gap between observations, i.e. the largest
safety margin on both sides."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from gazette.application.ask import AskQuestion
from gazette.application.ports import Judge
from gazette.domain.models import AnswerStatus, EvalItem
from gazette.domain.validity import ValidityCalendar

DEFAULT_AS_OF = date(2026, 11, 14)
RECALL_K = 5
MAX_FALSE_REFUSAL = 0.15


@dataclass(frozen=True)
class ItemResult:
    id: str
    should_refuse: bool
    explicit_as_of: bool
    gold: tuple[str, ...]
    top_similarity: float
    bm25: tuple[str, ...]
    vector: tuple[str, ...]
    hybrid: tuple[str, ...]
    reranked: tuple[str, ...]
    refused: bool
    cited: tuple[str, ...]
    not_in_force_cited: tuple[str, ...]
    groundedness: float | None


class EvaluateSystem:
    def __init__(self, ask: AskQuestion, calendar: ValidityCalendar, judge: Judge) -> None:
        self._ask = ask
        self._calendar = calendar
        self._judge = judge

    def evaluate_item(self, item: EvalItem) -> ItemResult:
        as_of = item.as_of or DEFAULT_AS_OF
        retrieval = self._ask.retrieve(item.question, as_of)
        answer = self._ask(item.question, as_of)
        refused = answer.status is AnswerStatus.REFUSED
        cited = tuple(c.chunk_id for c in answer.citations)
        return ItemResult(
            id=item.id,
            should_refuse=item.should_refuse,
            explicit_as_of=item.as_of is not None,
            gold=item.gold,
            top_similarity=retrieval.top_similarity,
            bm25=tuple(h.chunk_id for h in retrieval.keyword[:RECALL_K]),
            vector=tuple(h.chunk_id for h in retrieval.vector[:RECALL_K]),
            hybrid=tuple(h.chunk_id for h in retrieval.fused[:RECALL_K]),
            reranked=tuple(h.chunk_id for h in retrieval.hits[:RECALL_K]),
            refused=refused,
            cited=cited,
            not_in_force_cited=tuple(c for c in cited if not self._calendar.is_in_force(c, as_of)),
            groundedness=None if refused else self._judge.groundedness(answer.answer, answer.citations),
        )

    def run(self, items: Sequence[EvalItem]) -> dict:
        results = [self.evaluate_item(item) for item in items]
        answerable = [r for r in results if not r.should_refuse]
        refusable = [r for r in results if r.should_refuse]
        as_of_items = [r for r in results if r.explicit_as_of]
        citations = [(r, c) for r in results for c in r.cited]
        grounded = [r.groundedness for r in results if r.groundedness is not None]
        retrievers = ("bm25", "vector", "hybrid") + (("reranked",) if self._ask.reranker else ())
        return {
            "counts": {
                "items": len(results),
                "answerable": len(answerable),
                "should_refuse": len(refusable),
                "as_of": len(as_of_items),
            },
            "recall_at_5": {
                name: _round(_mean(_recall(getattr(r, name), r.gold) for r in answerable))
                for name in retrievers
            },
            "citation_precision": _round(_mean(c in r.gold for r, c in citations)),
            "refusal_recall": _round(_mean(r.refused for r in refusable)),
            "false_refusal_rate": _round(_mean(r.refused for r in answerable)),
            "as_of_accuracy": _round(_mean(_as_of_correct(r) for r in as_of_items)),
            "superseded_citations": sum(len(r.not_in_force_cited) for r in results),
            "groundedness": _round(_mean(grounded)),
            "threshold_sweep": sweep_threshold(
                [r.top_similarity for r in answerable], [r.top_similarity for r in refusable]
            ),
            "failures": _failures(results),
        }


def sweep_threshold(answerable: Sequence[float], refusable: Sequence[float],
                    max_false_refusal: float = MAX_FALSE_REFUSAL) -> dict:
    values = sorted(set(answerable) | set(refusable))
    candidates = [0.0] + [(a + b) / 2 for a, b in zip(values, values[1:])]
    rows = []
    for threshold in candidates:
        refusal_recall = _mean(s < threshold for s in refusable)
        false_refusal = _mean(s < threshold for s in answerable)
        if false_refusal > max_false_refusal:
            continue
        margin = min((abs(v - threshold) for v in values), default=0.0)
        rows.append((refusal_recall, margin, threshold, false_refusal))
    best = max(rows, key=lambda row: (row[0], row[1]))
    return {
        "selected": _round(best[2]),
        "refusal_recall": _round(best[0]),
        "false_refusal_rate": _round(best[3]),
        "max_false_refusal_budget": max_false_refusal,
        "margin": _round(best[1]),
    }


def _recall(retrieved: Sequence[str], gold: Sequence[str]) -> float:
    return len(set(retrieved) & set(gold)) / len(gold) if gold else 0.0


def _as_of_correct(result: ItemResult) -> bool:
    if result.should_refuse:
        return result.refused
    return (
        not result.refused
        and bool(set(result.cited) & set(result.gold))
        and not result.not_in_force_cited
    )


def _failures(results: Sequence[ItemResult]) -> list[dict]:
    failures = []
    for r in results:
        problem = None
        if r.should_refuse and not r.refused:
            problem = "answered but should refuse"
        elif not r.should_refuse and r.refused:
            problem = "refused but answerable"
        elif not r.should_refuse and not set(r.cited) & set(r.gold):
            problem = "no gold paragraph cited"
        elif r.not_in_force_cited:
            problem = "cited a paragraph not in force"
        if problem:
            failures.append({"id": r.id, "problem": problem, "gold": list(r.gold),
                             "cited": list(r.cited), "top_similarity": _round(r.top_similarity)})
    return failures


def _mean(values) -> float:
    values = [float(v) for v in values]
    return sum(values) / len(values) if values else 0.0


def _round(value: float) -> float:
    return round(value, 4)
