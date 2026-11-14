"""BM25 keyword index over paragraph chunks (rank_bm25.BM25Okapi)."""

from __future__ import annotations

import json
from collections.abc import Collection, Sequence
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from gazette.domain.models import Paragraph, ScoredChunk
from gazette.infrastructure.text import tokenize

_FILE = "bm25.json"


class BM25Index:
    def __init__(self) -> None:
        self._ids: list[str] = []
        self._tokens: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    def build(self, paragraphs: Sequence[Paragraph]) -> None:
        self._ids = [p.chunk_id for p in paragraphs]
        self._tokens = [tokenize(p.text) for p in paragraphs]
        self._bm25 = BM25Okapi(self._tokens)

    def save(self, directory: Path) -> None:
        payload = {"ids": self._ids, "tokens": self._tokens}
        (Path(directory) / _FILE).write_text(json.dumps(payload), encoding="utf-8")

    def load(self, directory: Path) -> None:
        payload = json.loads((Path(directory) / _FILE).read_text(encoding="utf-8"))
        self._ids, self._tokens = payload["ids"], payload["tokens"]
        self._bm25 = BM25Okapi(self._tokens)

    def search(
        self, query: str, k: int, allowed_ids: Collection[str] | None = None
    ) -> list[ScoredChunk]:
        if self._bm25 is None:
            raise RuntimeError("BM25 index not built or loaded")
        terms = tokenize(query)
        if not terms:
            return []
        scores = self._bm25.get_scores(terms)
        hits: list[ScoredChunk] = []
        for i in np.argsort(-scores, kind="stable"):
            if scores[i] <= 0:
                break
            if allowed_ids is None or self._ids[i] in allowed_ids:
                hits.append(ScoredChunk(self._ids[i], float(scores[i])))
                if len(hits) == k:
                    break
        return hits
