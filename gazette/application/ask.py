"""Ask use case: as-of filtering, hybrid retrieval, generation, citation validation, refusal.

The refusal decision is made here, independently of the LLM provider:
  1. no in-force paragraph is retrieved, or the best vector similarity is below
     ``min_similarity`` (threshold chosen from the eval set), or
  2. the provider returns no citation that survives validation.

If any citation was dropped, or the judge finds a draft sentence not supported by the
surviving quotes, the answer text is rebuilt from those verified quotes only, so an
unsupported claim can never ride along with one genuine citation."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import replace
from datetime import date

from gazette.application.ports import Judge, LLMProvider, Reranker
from gazette.application.retrieval import HybridRetriever, Retrieval
from gazette.domain.citations import validate_citations
from gazette.domain.models import Answer, AnswerStatus, Paragraph
from gazette.domain.validity import ValidityCalendar

NOT_COVERED = "not covered by the indexed circulars"
REASON_LOW_RELEVANCE = "no in-force paragraph is sufficiently relevant to the question"
REASON_NO_VALID_CITATION = "no citation could be verified against the retrieved paragraphs"


class AskQuestion:
    def __init__(
        self,
        retriever: HybridRetriever,
        provider: LLMProvider,
        judge: Judge,
        calendar: ValidityCalendar,
        paragraphs: Mapping[str, Paragraph],
        top_k: int,
        min_similarity: float,
        reranker: Reranker | None = None,
    ) -> None:
        self._retriever = retriever
        self._provider = provider
        self._judge = judge
        self._calendar = calendar
        self._paragraphs = paragraphs
        self._top_k = top_k
        self._min_similarity = min_similarity
        self._reranker = reranker

    @property
    def provider(self) -> LLMProvider:
        return self._provider

    @property
    def reranker(self) -> Reranker | None:
        return self._reranker

    def retrieve(self, question: str, as_of: date) -> Retrieval:
        """In-force hybrid retrieval; with a reranker, the fused pool is re-scored and
        the reranker's top-k replaces the RRF top-k."""
        allowed = self._calendar.in_force_ids(as_of)
        retrieval = self._retriever.retrieve(question, self._top_k, allowed)
        if self._reranker is None or not retrieval.fused:
            return retrieval
        pool = [self._paragraphs[h.chunk_id] for h in retrieval.fused]
        reranked = self._reranker.rerank(question, pool)
        return replace(retrieval, hits=tuple(reranked[: self._top_k]))

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
        answer = draft.answer
        if check.rejected or self._judge.groundedness(answer, check.valid) < 1.0:
            answer = " ".join(c.quote for c in check.valid)
        return Answer(
            query_id=query_id,
            status=AnswerStatus.ANSWERED,
            answer=answer,
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
