"""Ingest use case: load corpus, check invariants, build and persist both indexes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from gazette.application.ports import CorpusSource, SearchIndex
from gazette.domain.corpus import all_paragraphs, check_integrity

MANIFEST_FILE = "manifest.json"


@dataclass(frozen=True)
class IngestReport:
    circulars: int
    paragraphs: int
    amending_circulars: int
    corpus_fingerprint: str


class IngestCorpus:
    def __init__(self, corpus: CorpusSource, keyword: SearchIndex, vector: SearchIndex) -> None:
        self._corpus = corpus
        self._indexes = (keyword, vector)

    def run(self, index_dir: Path) -> IngestReport:
        circulars = self._corpus.load()
        check_integrity(circulars)
        paragraphs = all_paragraphs(circulars)
        index_dir.mkdir(parents=True, exist_ok=True)
        for index in self._indexes:
            index.build(paragraphs)
            index.save(index_dir)
        report = IngestReport(
            circulars=len(circulars),
            paragraphs=len(paragraphs),
            amending_circulars=sum(1 for c in circulars if c.amends),
            corpus_fingerprint=self._corpus.fingerprint(),
        )
        (index_dir / MANIFEST_FILE).write_text(json.dumps(asdict(report), indent=2), "utf-8")
        return report


def read_manifest(index_dir: Path) -> IngestReport:
    path = index_dir / MANIFEST_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m gazette.ingest` first")
    return IngestReport(**json.loads(path.read_text("utf-8")))
