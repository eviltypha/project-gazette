"""Feedback use cases: submit, export candidates, approve one candidate into the eval set.

Nothing reaches eval/questions.jsonl without an explicit ``approve`` of one candidate."""

from __future__ import annotations

from gazette.application.ports import AuditRepository, EvalSetRepository, FeedbackRepository
from gazette.domain.models import AuditRecord, EvalItem, Feedback, Verdict


class UnknownQueryError(LookupError):
    pass


class CandidateNotFoundError(LookupError):
    pass


class SubmitFeedback:
    def __init__(self, audit: AuditRepository, feedback: FeedbackRepository) -> None:
        self._audit = audit
        self._feedback = feedback

    def __call__(self, feedback: Feedback) -> int:
        if self._audit.get(feedback.query_id) is None:
            raise UnknownQueryError(feedback.query_id)
        return self._feedback.add(feedback)


def candidate_id(feedback_id: int) -> str:
    return f"fb-{feedback_id:05d}"


def to_candidate(feedback_id: int, feedback: Feedback, record: AuditRecord) -> EvalItem:
    should_refuse = feedback.verdict is Verdict.SHOULD_HAVE_REFUSED
    return EvalItem(
        id=candidate_id(feedback_id),
        question=record.question,
        as_of=record.as_of,
        gold=() if should_refuse else feedback.correct_citations,
        should_refuse=should_refuse,
    )


class ExportCandidates:
    def __init__(
        self,
        audit: AuditRepository,
        feedback: FeedbackRepository,
        candidates: EvalSetRepository,
    ) -> None:
        self._audit = audit
        self._feedback = feedback
        self._candidates = candidates

    def __call__(self) -> list[EvalItem]:
        items = []
        for feedback_id, feedback in self._feedback.list_actionable():
            record = self._audit.get(feedback.query_id)
            if record is not None:
                items.append(to_candidate(feedback_id, feedback, record))
        self._candidates.write(items)
        return items


class ApproveCandidate:
    def __init__(self, candidates: EvalSetRepository, questions: EvalSetRepository) -> None:
        self._candidates = candidates
        self._questions = questions

    def __call__(self, item_id: str) -> EvalItem:
        pending = self._candidates.load()
        match = next((item for item in pending if item.id == item_id), None)
        if match is None:
            raise CandidateNotFoundError(item_id)
        self._questions.append(match)
        self._candidates.write([item for item in pending if item.id != item_id])
        return match
