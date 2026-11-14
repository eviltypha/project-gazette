"""FastAPI delivery layer. Validates payloads and maps them to use cases."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import date
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from gazette.config import Settings
from gazette.container import Services, build_services
from gazette.domain.models import Answer

Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=1000)]


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: Question
    as_of: date | None = Field(default=None, description="Defaults to today")


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


def create_app(settings_factory: Callable[[], Settings] = Settings.from_env) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.services = build_services(settings_factory())
        yield

    app = FastAPI(title="Project Gazette", version="0.1.0", lifespan=lifespan)

    def services(request: Request) -> Services:
        return request.app.state.services

    @app.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest, svc: Annotated[Services, Depends(services)]) -> AskResponse:
        as_of = body.as_of or date.today()
        return AskResponse.from_answer(svc.ask(body.question, as_of), as_of)

    return app
