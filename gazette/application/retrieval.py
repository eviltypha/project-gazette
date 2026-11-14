"""Hybrid retrieval: BM25 + vector results fused with Reciprocal Rank Fusion (RRF).

RRF score(d) = sum over retrievers of 1 / (rrf_k + rank_r(d)), rank starting at 1.
It needs no score normalisation across retrievers, and a paragraph found by only
one retriever still receives credit from that retriever's rank."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Sequence
from dataclasses import dataclass

from gazette.application.ports import SearchIndex
from gazette.domain.models import ScoredChunk

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[ScoredChunk]], rrf_k: int = DEFAULT_RRF_K
) -> list[ScoredChunk]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.chunk_id] += 1.0 / (rrf_k + rank)
    return [
        ScoredChunk(cid, score)
        for cid, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    ]


@dataclass(frozen=True)
class Retrieval:
    hits: tuple[ScoredChunk, ...]
    keyword: tuple[ScoredChunk, ...]
    vector: tuple[ScoredChunk, ...]
    fused: tuple[ScoredChunk, ...]

    @property
    def top_similarity(self) -> float:
        """Best cosine similarity from the vector retriever; used for refusal."""
        return self.vector[0].score if self.vector else 0.0


class HybridRetriever:
    def __init__(
        self,
        keyword: SearchIndex,
        vector: SearchIndex,
        candidates: int = 20,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        self.keyword = keyword
        self.vector = vector
        self._candidates = candidates
        self._rrf_k = rrf_k

    def retrieve(
        self, query: str, k: int, allowed_ids: Collection[str] | None = None
    ) -> Retrieval:
        keyword_hits = self.keyword.search(query, self._candidates, allowed_ids)
        vector_hits = self.vector.search(query, self._candidates, allowed_ids)
        fused = reciprocal_rank_fusion([keyword_hits, vector_hits], self._rrf_k)
        return Retrieval(tuple(fused[:k]), tuple(keyword_hits), tuple(vector_hits), tuple(fused))
