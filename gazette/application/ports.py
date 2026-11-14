"""Ports: abstract interfaces implemented by infrastructure adapters."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from gazette.domain.models import (
    AuditRecord,
    Circular,
    Citation,
    EvalItem,
    Feedback,
    Paragraph,
    ScoredChunk,
)


class CorpusSource(Protocol):
    def load(self) -> list[Circular]: ...

    def fingerprint(self) -> str:
        """Stable hash of the corpus content; changes whenever any circular changes."""
        ...


class Embedder(Protocol):
    name: str

    def fit(self, texts: Sequence[str]) -> None:
        """Learn corpus statistics (no-op for pretrained models)."""
        ...

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (n, d) matrix of L2-normalised vectors."""
        ...

    def state(self) -> dict: ...

    def load_state(self, state: dict) -> None: ...


class SearchIndex(Protocol):
    """A single retriever (keyword or vector) over paragraph chunks."""

    def build(self, paragraphs: Sequence[Paragraph]) -> None: ...

    def save(self, directory: Path) -> None: ...

    def load(self, directory: Path) -> None: ...

    def search(
        self, query: str, k: int, allowed_ids: Collection[str] | None = None
    ) -> list[ScoredChunk]: ...


@dataclass(frozen=True)
class Draft:
    """Unvalidated model output; citations are checked by the pipeline."""

    answer: str
    citations: tuple[Citation, ...]


class Reranker(Protocol):
    """Re-scores a candidate pool by reading question and paragraph together."""

    name: str

    def rerank(self, query: str, paragraphs: Sequence[Paragraph]) -> list[ScoredChunk]:
        """Return all candidates ordered best-first with the reranker's scores."""
        ...


class LLMProvider(Protocol):
    name: str
    prompt_version: str

    def generate(self, question: str, context: Sequence[Paragraph]) -> Draft: ...


class Judge(Protocol):
    def groundedness(self, answer: str, citations: Sequence[Citation]) -> float:
        """Return a score in [0, 1]."""
        ...


class AuditRepository(Protocol):
    def record(self, entry: AuditRecord) -> None: ...

    def get(self, query_id: str) -> AuditRecord | None: ...


class FeedbackRepository(Protocol):
    def add(self, feedback: Feedback) -> int:
        """Persist feedback and return its id."""
        ...

    def list_actionable(self) -> list[tuple[int, Feedback]]:
        """All (id, feedback) pairs whose verdict is not ``correct``."""
        ...


class EvalSetRepository(Protocol):
    def load(self) -> list[EvalItem]: ...

    def append(self, item: EvalItem) -> None: ...

    def write(self, items: list[EvalItem]) -> None: ...
