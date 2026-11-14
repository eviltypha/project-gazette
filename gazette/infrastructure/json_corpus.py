"""Loads circulars from JSON files (one circular per file) in a directory."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

from gazette.domain.models import Amendment, Circular, Paragraph


class CorpusError(ValueError):
    """Raised when a circular file is missing required fields or is malformed."""


class JsonCorpusSource:
    def __init__(self, directory: Path) -> None:
        self._directory = Path(directory)

    def _files(self) -> list[Path]:
        files = sorted(self._directory.glob("*.json"))
        if not files:
            raise CorpusError(f"no circular files found in {self._directory}")
        return files

    def load(self) -> list[Circular]:
        return [self._parse(path) for path in self._files()]

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for path in self._files():
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()

    @staticmethod
    def _parse(path: Path) -> Circular:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            circular_id = raw["circular_id"]
            return Circular(
                circular_id=circular_id,
                title=raw["title"],
                issue_date=date.fromisoformat(raw["issue_date"]),
                effective_date=date.fromisoformat(raw["effective_date"]),
                paragraphs=tuple(
                    Paragraph(circular_id, str(p["para"]), p["text"]) for p in raw["paragraphs"]
                ),
                amends=tuple(
                    Amendment(a["circular_id"], tuple(str(n) for n in a["paras"]))
                    for a in raw.get("amends", [])
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CorpusError(f"{path.name}: {exc!r}") from exc
