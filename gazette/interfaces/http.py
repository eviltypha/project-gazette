"""FastAPI delivery layer. Validates payloads and maps them to use cases."""

from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Request, status
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, StringConstraints

from gazette.application.feedback import UnknownQueryError
from gazette.config import Settings
from gazette.container import Services, build_services
from gazette.domain.models import Answer, AuditRecord, Feedback, Verdict

IST = timezone(timedelta(hours=5, minutes=30))
CHUNK_ID_PATTERN = r"^[A-Z0-9][A-Z0-9-]*#\d+$"

def _iso_date_string(value: object) -> object:
    if not isinstance(value, str):
        raise ValueError("as_of must be a YYYY-MM-DD string")
    return date.fromisoformat(value)


Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=1000)]
ChunkId = Annotated[str, StringConstraints(pattern=CHUNK_ID_PATTERN, max_length=64)]
IsoDate = Annotated[date, BeforeValidator(_iso_date_string)]


def today_ist() -> date:
    """'Today' for an Indian bank is the IST calendar date, not the host's."""
    return datetime.now(IST).date()


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: Question
    as_of: IsoDate | None = Field(default=None, description="YYYY-MM-DD; defaults to today (IST)")


class CitationOut(BaseModel):
    circular_id: str
    para: str
    quote: str


class AskResponse(BaseModel):
    query_id: str
    status: Literal["answered", "refused"]
    answer: str
    citations: list[CitationOut]
    retrieved: list[str]
    as_of: date
    refusal_reason: str | None = None

    @classmethod
    def from_answer(cls, answer: Answer, as_of: date) -> "AskResponse":
        return cls(
            query_id=answer.query_id,
            status=answer.status.value,
            answer=answer.answer,
            citations=[CitationOut(**vars(c)) for c in answer.citations],
            retrieved=[h.chunk_id for h in answer.retrieved],
            as_of=as_of,
            refusal_reason=answer.refusal_reason,
        )


class ScoredChunkOut(BaseModel):
    chunk_id: str
    score: float


class AuditResponse(BaseModel):
    query_id: str
    timestamp: str
    question: str
    as_of: date
    retrieved: list[ScoredChunkOut]
    provider: str
    prompt_version: str
    status: Literal["answered", "refused"]
    citations: list[CitationOut]
    latency_ms: float
    refusal_reason: str | None

    @classmethod
    def from_record(cls, record: AuditRecord) -> "AuditResponse":
        return cls(
            **{k: v for k, v in vars(record).items() if k not in {"retrieved", "citations", "status"}},
            status=record.status.value,
            retrieved=[ScoredChunkOut(**vars(h)) for h in record.retrieved],
            citations=[CitationOut(**vars(c)) for c in record.citations],
        )


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query_id: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]
    verdict: Verdict
    correct_citations: list[ChunkId] = Field(default_factory=list, max_length=10)
    reviewer: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
    comment: Annotated[str, StringConstraints(max_length=2000)] | None = None


class FeedbackResponse(BaseModel):
    feedback_id: int
    query_id: str


def create_app(settings_factory: Callable[[], Settings] = Settings.from_env) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.services = build_services(settings_factory())
        yield

    app = FastAPI(title="Project Gazette", version="0.1.0", lifespan=lifespan)

    def services(request: Request) -> Services:
        return request.app.state.services

    ServicesDep = Annotated[Services, Depends(services)]

    @app.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest, svc: ServicesDep) -> AskResponse:
        as_of = body.as_of or today_ist()
        return AskResponse.from_answer(svc.audited_ask(body.question, as_of), as_of)

    @app.get("/audit/{query_id}", response_model=AuditResponse)
    def audit(query_id: str, svc: ServicesDep) -> AuditResponse:
        record = svc.audit.get(query_id)
        if record is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown query_id {query_id!r}")
        return AuditResponse.from_record(record)

    @app.post("/feedback", response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED)
    def feedback(body: FeedbackRequest, svc: ServicesDep) -> FeedbackResponse:
        try:
            feedback_id = svc.submit_feedback(
                Feedback(
                    query_id=body.query_id,
                    verdict=body.verdict,
                    reviewer=body.reviewer,
                    correct_citations=tuple(body.correct_citations),
                    comment=body.comment,
                )
            )
        except UnknownQueryError:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown query_id {body.query_id!r}")
        return FeedbackResponse(feedback_id=feedback_id, query_id=body.query_id)

    return app
