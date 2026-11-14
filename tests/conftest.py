"""Shared fixtures. Tests never rely on a committed index/ or a network."""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "data" / "circulars"


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch, tmp_path):
    monkeypatch.setenv("GAZETTE_CORPUS_DIR", str(CORPUS_DIR))
    monkeypatch.setenv("GAZETTE_INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("GAZETTE_DB_PATH", str(tmp_path / "gazette.sqlite3"))
    monkeypatch.setenv("GAZETTE_LLM_PROVIDER", "fake")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
