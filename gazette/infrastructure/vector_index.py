"""In-memory NumPy vector store with cosine similarity, persisted to disk."""

from __future__ import annotations

import json
from collections.abc import Collection, Sequence
from pathlib import Path

import numpy as np

from gazette.application.ports import Embedder
from gazette.domain.models import Paragraph, ScoredChunk

_VECTORS = "vectors.npy"
_META = "vectors.json"


class NumpyVectorIndex:
    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._ids: list[str] = []
        self._matrix: np.ndarray | None = None

    def build(self, paragraphs: Sequence[Paragraph]) -> None:
        texts = [p.text for p in paragraphs]
        self._embedder.fit(texts)
        self._ids = [p.chunk_id for p in paragraphs]
        self._matrix = self._embedder.embed(texts)

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        np.save(directory / _VECTORS, self._matrix)
        meta = {"ids": self._ids, "embedder": self._embedder.name, "state": self._embedder.state()}
        (directory / _META).write_text(json.dumps(meta), encoding="utf-8")

    def load(self, directory: Path) -> None:
        directory = Path(directory)
        meta = json.loads((directory / _META).read_text(encoding="utf-8"))
        if meta["embedder"] != self._embedder.name:
            raise RuntimeError(
                f"index built with embedder {meta['embedder']!r}, "
                f"configured {self._embedder.name!r}; re-run ingest"
            )
        self._embedder.load_state(meta["state"])
        self._ids = meta["ids"]
        self._matrix = np.load(directory / _VECTORS)

    def search(
        self, query: str, k: int, allowed_ids: Collection[str] | None = None
    ) -> list[ScoredChunk]:
        if self._matrix is None:
            raise RuntimeError("vector index not built or loaded")
        sims = self._matrix @ self._embedder.embed([query])[0]
        hits: list[ScoredChunk] = []
        for i in np.argsort(-sims, kind="stable"):
            if sims[i] <= 0:
                break
            if allowed_ids is None or self._ids[i] in allowed_ids:
                hits.append(ScoredChunk(self._ids[i], float(sims[i])))
                if len(hits) == k:
                    break
        return hits
