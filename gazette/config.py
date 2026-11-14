"""Runtime settings read from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    corpus_dir: Path
    index_dir: Path
    db_path: Path
    llm_provider: str
    embedder: str
    top_k: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            corpus_dir=Path(os.getenv("GAZETTE_CORPUS_DIR", "data/circulars")),
            index_dir=Path(os.getenv("GAZETTE_INDEX_DIR", "index")),
            db_path=Path(os.getenv("GAZETTE_DB_PATH", "gazette.sqlite3")),
            llm_provider=os.getenv("GAZETTE_LLM_PROVIDER", "fake"),
            embedder=os.getenv("GAZETTE_EMBEDDER", "hashed"),
            top_k=int(os.getenv("GAZETTE_TOP_K", "5")),
        )
