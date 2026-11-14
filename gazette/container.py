"""Composition root: the only place that knows concrete adapters."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

from gazette.application.ask import AskQuestion
from gazette.application.audit import AuditedAsk
from gazette.application.evaluation import EvaluateSystem
from gazette.application.feedback import ApproveCandidate, ExportCandidates, SubmitFeedback
from gazette.application.ingest import IngestCorpus, IngestReport, read_manifest
from gazette.application.ports import Embedder, LLMProvider, Reranker
from gazette.application.retrieval import HybridRetriever
from gazette.config import Settings
from gazette.domain.corpus import all_paragraphs
from gazette.domain.models import Circular, EvalItem
from gazette.domain.validity import ValidityCalendar
from gazette.infrastructure.bm25_index import BM25Index
from gazette.infrastructure.embedders import HashedTfidfEmbedder
from gazette.infrastructure.fake_provider import FakeProvider
from gazette.infrastructure.json_corpus import JsonCorpusSource
from gazette.infrastructure.jsonl_eval_set import JsonlEvalSet
from gazette.infrastructure.overlap_judge import OverlapJudge
from gazette.infrastructure.sqlite_store import SqliteAuditRepository, SqliteFeedbackRepository
from gazette.infrastructure.vector_index import NumpyVectorIndex


def make_embedder(settings: Settings) -> Embedder:
    if settings.embedder == "hashed":
        return HashedTfidfEmbedder()
    raise ValueError(f"unknown GAZETTE_EMBEDDER {settings.embedder!r}")


def make_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "fake":
        return FakeProvider()
    raise ValueError(f"unknown GAZETTE_LLM_PROVIDER {settings.llm_provider!r}")


def make_reranker(settings: Settings) -> Reranker | None:
    if settings.reranker == "none":
        return None
    if settings.reranker == "cross-encoder":
        from gazette.infrastructure.cross_encoder import CrossEncoderReranker

        return CrossEncoderReranker()
    raise ValueError(f"unknown GAZETTE_RERANKER {settings.reranker!r}")


def run_ingest(settings: Settings) -> IngestReport:
    use_case = IngestCorpus(
        JsonCorpusSource(settings.corpus_dir),
        BM25Index(),
        NumpyVectorIndex(make_embedder(settings)),
    )
    return use_case.run(settings.index_dir)


@dataclass(frozen=True)
class Services:
    settings: Settings
    circulars: tuple[Circular, ...]
    calendar: ValidityCalendar
    ask: AskQuestion
    audited_ask: AuditedAsk
    audit: SqliteAuditRepository
    submit_feedback: SubmitFeedback


def build_services(settings: Settings) -> Services:
    corpus = JsonCorpusSource(settings.corpus_dir)
    manifest = read_manifest(settings.index_dir)
    if manifest.corpus_fingerprint != corpus.fingerprint():
        raise RuntimeError("corpus changed since the index was built; re-run ingest")
    circulars = tuple(corpus.load())

    keyword, vector = BM25Index(), NumpyVectorIndex(make_embedder(settings))
    keyword.load(settings.index_dir)
    vector.load(settings.index_dir)

    calendar = ValidityCalendar(circulars)
    paragraphs = {p.chunk_id: p for p in all_paragraphs(circulars)}
    ask = AskQuestion(
        retriever=HybridRetriever(keyword, vector),
        provider=make_provider(settings),
        judge=OverlapJudge(),
        calendar=calendar,
        paragraphs=paragraphs,
        top_k=settings.top_k,
        min_similarity=settings.min_similarity,
        reranker=make_reranker(settings),
    )
    audit = SqliteAuditRepository(settings.db_path)
    return Services(
        settings=settings,
        circulars=circulars,
        calendar=calendar,
        ask=ask,
        audited_ask=AuditedAsk(ask, audit),
        audit=audit,
        submit_feedback=SubmitFeedback(
            audit, SqliteFeedbackRepository(settings.db_path), frozenset(paragraphs)
        ),
    )


def export_candidates(settings: Settings, candidates: Path) -> list[EvalItem]:
    return ExportCandidates(
        SqliteAuditRepository(settings.db_path),
        SqliteFeedbackRepository(settings.db_path),
        JsonlEvalSet(candidates),
    )()


def approve_candidate(candidates: Path, questions: Path, item_id: str) -> EvalItem:
    return ApproveCandidate(JsonlEvalSet(candidates), JsonlEvalSet(questions))(item_id)


def run_evaluation(settings: Settings, questions: Path) -> dict:
    """Evaluate with the FakeProvider against a fresh index built in a temp directory,
    so the report depends only on the committed corpus, eval set and code."""
    with tempfile.TemporaryDirectory() as tmp:
        eval_settings = replace(settings, index_dir=Path(tmp) / "index", llm_provider="fake")
        run_ingest(eval_settings)
        services = build_services(eval_settings)
        evaluator = EvaluateSystem(services.ask, services.calendar, OverlapJudge())
        report = evaluator.run(JsonlEvalSet(questions).load())
    report["config"] = {
        "provider": services.ask.provider.name,
        "prompt_version": services.ask.provider.prompt_version,
        "embedder": eval_settings.embedder,
        "top_k": eval_settings.top_k,
        "min_similarity": eval_settings.min_similarity,
        "reranker": eval_settings.reranker,
    }
    return report