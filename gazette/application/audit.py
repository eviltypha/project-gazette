"""Audit use case: wraps the ask pipeline and records every call."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, date, datetime

from gazette.application.ask import AskQuestion
from gazette.application.ports import AuditRepository
from gazette.domain.models import Answer, AuditRecord


class AuditedAsk:
    def __init__(
        self,
        ask: AskQuestion,
        audit: AuditRepository,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._ask = ask
        self._audit = audit
        self._clock = clock

    def __call__(self, question: str, as_of: date) -> Answer:
        timestamp = self._clock()
        started = time.perf_counter()
        answer = self._ask(question, as_of)
        latency_ms = (time.perf_counter() - started) * 1000
        self._audit.record(
            AuditRecord(
                query_id=answer.query_id,
                timestamp=timestamp.isoformat(),
                question=question,
                as_of=as_of,
                retrieved=answer.retrieved,
                provider=self._ask.provider.name,
                prompt_version=self._ask.provider.prompt_version,
                status=answer.status,
                citations=answer.citations,
                latency_ms=round(latency_ms, 3),
                refusal_reason=answer.refusal_reason,
            )
        )
        return answer
