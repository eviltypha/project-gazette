from gazette.application.retrieval import HybridRetriever, reciprocal_rank_fusion
from gazette.domain.models import ScoredChunk
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
