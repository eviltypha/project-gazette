"""Small builders shared by unit tests."""

from __future__ import annotations

import json
from collections.abc import Collection, Sequence
from pathlib import Path

from gazette.domain.models import ScoredChunk


def write_circular(directory: Path, circular_id: str, effective: str, paras: list[str],
                   amends: list[dict] | None = None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "circular_id": circular_id,
        "title": f"Title of {circular_id}",
        "issue_date": effective,
        "effective_date": effective,
        "amends": amends or [],
        "paragraphs": [{"para": str(i), "text": t} for i, t in enumerate(paras, start=1)],
    }
    (directory / f"{circular_id}.json").write_text(json.dumps(payload), encoding="utf-8")


class StubIndex:
    """SearchIndex returning a fixed ranking, filtered by allowed ids."""

    def __init__(self, ranking: Sequence[str]) -> None:
        self._ranking = list(ranking)

    def search(self, query: str, k: int, allowed_ids: Collection[str] | None = None):
        ids = [c for c in self._ranking if allowed_ids is None or c in allowed_ids]
        return [ScoredChunk(c, 1.0 - i * 0.1) for i, c in enumerate(ids[:k])]
