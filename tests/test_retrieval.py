from datetime import date

from gazette.application.ask import AskQuestion
from gazette.application.retrieval import HybridRetriever, reciprocal_rank_fusion
from gazette.domain.models import Circular, Paragraph, ScoredChunk
from gazette.domain.validity import ValidityCalendar
from gazette.infrastructure.fake_provider import FakeProvider
from gazette.infrastructure.overlap_judge import OverlapJudge
from tests.helpers import StubIndex


def _ranking(*ids):
    return [ScoredChunk(i, 1.0) for i in ids]


def test_rrf_ranks_paragraph_found_by_only_one_retriever():
    fused = reciprocal_rank_fusion([_ranking("A", "B"), _ranking("C", "A")])

    ids = [h.chunk_id for h in fused]
    assert ids == ["A", "C", "B"]  # C (vector-only, rank 1) beats B (keyword-only, rank 2)
    assert fused[1].score == 1 / 61


def test_hybrid_retriever_returns_top_k_restricted_to_allowed_ids():
    retriever = HybridRetriever(StubIndex(["A", "B", "X"]), StubIndex(["X", "C", "A"]))

    result = retriever.retrieve("q", k=2, allowed_ids={"A", "B", "C"})

    assert [h.chunk_id for h in result.hits] == ["A", "C"]
    assert all(h.chunk_id != "X" for h in result.keyword + result.vector)


class ReverseReranker:
    name = "reverse"

    def rerank(self, query, paragraphs):
        return [ScoredChunk(p.chunk_id, float(i)) for i, p in enumerate(reversed(paragraphs))]


def test_reranker_reorders_fused_pool_and_keeps_top_k_in_force():
    paras = tuple(Paragraph("GZ-A", str(n), f"text {n}") for n in range(1, 5))
    ask = AskQuestion(
        retriever=HybridRetriever(StubIndex(["GZ-A#1", "GZ-A#2", "GZ-A#3", "GZ-A#4"]), StubIndex([])),
        provider=FakeProvider(),
        judge=OverlapJudge(),
        calendar=ValidityCalendar([Circular("GZ-A", "t", date(2020, 1, 1), date(2020, 1, 1), paras)]),
        paragraphs={p.chunk_id: p for p in paras},
        top_k=2,
        min_similarity=0.0,
        reranker=ReverseReranker(),
    )

    retrieval = ask.retrieve("q", date(2024, 1, 1))

    assert [h.chunk_id for h in retrieval.fused[:2]] == ["GZ-A#1", "GZ-A#2"]
    assert [h.chunk_id for h in retrieval.hits] == ["GZ-A#4", "GZ-A#3"]
