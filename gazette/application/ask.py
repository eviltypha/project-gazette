"""Ask use case: as-of filtering, hybrid retrieval, generation, citation validation, refusal.

The refusal decision is made here, independently of the LLM provider:
  1. no in-force paragraph is retrieved, or the best vector similarity is below
     ``min_similarity`` (threshold chosen from the eval set), or
  2. the provider returns no citation that survives validation."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from datetime import date

from gazette.application.ports import LLMProvider
from gazette.application.retrieval import HybridRetriever, Retrieval
from gazette.domain.citations import validate_citations
from gazette.domain.models import Answer, AnswerStatus, Paragraph
from gazette.domain.validity import ValidityCalendar

NOT_COVERED = "Not covered by the indexed circulars."
REASON_LOW_RELEVANCE = "no in-force paragraph is sufficiently relevant to the question"
REASON_NO_VALID_CITATION = "no citation could be verified against the retrieved paragraphs"


class AskQuestion:
    def __init__(
        self,
        retriever: HybridRetriever,
        provider: LLMProvider,
        calendar: ValidityCalendar,
        paragraphs: Mapping[str, Paragraph],
        top_k: int,
        min_similarity: float,
    ) -> None:
        self._retriever = retriever
        self._provider = provider
        self._calendar = calendar
        self._paragraphs = paragraphs
        self._top_k = top_k
        self._min_similarity = min_similarity

    @property
    def provider(self) -> LLMProvider:
        return self._provider

    def retrieve(self, question: str, as_of: date) -> Retrieval:
        allowed = self._calendar.in_force_ids(as_of)
        return self._retriever.retrieve(question, self._top_k, allowed)

    def __call__(self, question: str, as_of: date) -> Answer:
        query_id = uuid.uuid4().hex
        retrieval = self.retrieve(question, as_of)
        if not retrieval.hits or retrieval.top_similarity < self._min_similarity:
            return _refusal(query_id, retrieval, REASON_LOW_RELEVANCE)

        context = [self._paragraphs[h.chunk_id] for h in retrieval.hits]
        draft = self._provider.generate(question, context)
        check = validate_citations(
            draft.citations, {h.chunk_id for h in retrieval.hits}, self._paragraphs
        )
        if not check.valid:
            return _refusal(query_id, retrieval, REASON_NO_VALID_CITATION)
        return Answer(
            query_id=query_id,
            status=AnswerStatus.ANSWERED,
            answer=draft.answer,
            citations=check.valid,
            retrieved=retrieval.hits,
        )


def _refusal(query_id: str, retrieval: Retrieval, reason: str) -> Answer:
    return Answer(
        query_id=query_id,
        status=AnswerStatus.REFUSED,
        answer=NOT_COVERED,
        citations=(),
        retrieved=retrieval.hits,
        refusal_reason=reason,
    )
