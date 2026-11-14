"""Core entities. Plain immutable dataclasses with no framework or I/O dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum


def chunk_key(circular_id: str, para: str) -> str:
    return f"{circular_id}#{para}"


@dataclass(frozen=True)
class Amendment:
    """Paragraphs of an earlier circular replaced by the amending circular."""

    circular_id: str
    paras: tuple[str, ...]


@dataclass(frozen=True)
class Paragraph:
    circular_id: str
    para: str
    text: str

    @property
    def chunk_id(self) -> str:
        return chunk_key(self.circular_id, self.para)


@dataclass(frozen=True)
class Circular:
    circular_id: str
    title: str
    issue_date: date
    effective_date: date
    paragraphs: tuple[Paragraph, ...]
    amends: tuple[Amendment, ...] = ()


@dataclass(frozen=True)
class ScoredChunk:
    chunk_id: str
    score: float


@dataclass(frozen=True)
class Citation:
    circular_id: str
    para: str
    quote: str

    @property
    def chunk_id(self) -> str:
        return chunk_key(self.circular_id, self.para)


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    REFUSED = "refused"


@dataclass(frozen=True)
class Answer:
    query_id: str
    status: AnswerStatus
    answer: str
    citations: tuple[Citation, ...]
    retrieved: tuple[ScoredChunk, ...]
    refusal_reason: str | None = None


class Verdict(StrEnum):
    CORRECT = "correct"
    WRONG_ANSWER = "wrong_answer"
    WRONG_CITATION = "wrong_citation"
    SHOULD_HAVE_REFUSED = "should_have_refused"
    SHOULD_HAVE_ANSWERED = "should_have_answered"


@dataclass(frozen=True)
class Feedback:
    query_id: str
    verdict: Verdict
    reviewer: str
    correct_citations: tuple[str, ...] = ()
    comment: str | None = None


@dataclass(frozen=True)
class AuditRecord:
    query_id: str
    timestamp: str
    question: str
    as_of: date
    retrieved: tuple[ScoredChunk, ...]
    provider: str
    prompt_version: str
    status: AnswerStatus
    citations: tuple[Citation, ...]
    latency_ms: float
    refusal_reason: str | None = None


@dataclass(frozen=True)
class EvalItem:
    id: str
    question: str
    gold: tuple[str, ...]
    should_refuse: bool
    as_of: date | None = None
