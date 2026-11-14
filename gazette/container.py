"""Composition root: the only place that knows concrete adapters."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

from gazette.application.ask import AskQuestion
from gazette.application.evaluation import EvaluateSystem
from gazette.application.ingest import IngestCorpus, IngestReport, read_manifest
from gazette.application.ports import Embedder, LLMProvider
from gazette.application.retrieval import HybridRetriever
from gazette.config import Settings
from gazette.domain.corpus import all_paragraphs
from gazette.domain.models import Circular
from gazette.domain.validity import ValidityCalendar
from gazette.infrastructure.bm25_index import BM25Index
from gazette.infrastructure.embedders import HashedTfidfEmbedder
from gazette.infrastructure.fake_provider import FakeProvider
from gazette.infrastructure.json_corpus import JsonCorpusSource
from gazette.infrastructure.jsonl_eval_set import JsonlEvalSet
from gazette.infrastructure.overlap_judge import OverlapJudge
from gazette.infrastructure.vector_index import NumpyVectorIndex


def make_embedder(settings: Settings) -> Embedder:
    if settings.embedder == "hashed":
        return HashedTfidfEmbedder()
    raise ValueError(f"unknown GAZETTE_EMBEDDER {settings.embedder!r}")


def make_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "fake":
        return FakeProvider()
    raise ValueError(f"unknown GAZETTE_LLM_PROVIDER {settings.llm_provider!r}")


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
    ask = AskQuestion(
        retriever=HybridRetriever(keyword, vector),
        provider=make_provider(settings),
        calendar=calendar,
        paragraphs={p.chunk_id: p for p in all_paragraphs(circulars)},
        top_k=settings.top_k,
        min_similarity=settings.min_similarity,
    )
    return Services(settings, circulars, calendar, ask)


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
    }
    return report