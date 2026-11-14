"""Optional cross-encoder reranker (sentence-transformers). Off by default.

Needs ``pip install -e .[rerank]`` and a one-time model download, so it is never used
by pytest or the default eval; enable with ``GAZETTE_RERANKER=cross-encoder``."""

from __future__ import annotations

from collections.abc import Sequence

from gazette.domain.models import Paragraph, ScoredChunk

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        from sentence_transformers import CrossEncoder

        self.name = f"cross-encoder:{model_name}"
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, paragraphs: Sequence[Paragraph]) -> list[ScoredChunk]:
        if not paragraphs:
            return []
        scores = self._model.predict([(query, p.text) for p in paragraphs])
        ranked = sorted(zip(paragraphs, scores), key=lambda pair: -float(pair[1]))
        return [ScoredChunk(p.chunk_id, float(score)) for p, score in ranked]
