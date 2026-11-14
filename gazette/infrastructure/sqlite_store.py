"""SQLite-backed audit log and feedback store (stdlib sqlite3).

A short-lived connection per operation keeps the repositories safe to use from
FastAPI's worker threads."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path

from gazette.domain.models import (
    AnswerStatus,
    AuditRecord,
    Citation,
    Feedback,
    ScoredChunk,
    Verdict,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    query_id       TEXT PRIMARY KEY,
    timestamp      TEXT NOT NULL,
    question       TEXT NOT NULL,
    as_of          TEXT NOT NULL,
    retrieved      TEXT NOT NULL,
    provider       TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    status         TEXT NOT NULL,
    citations      TEXT NOT NULL,
    refusal_reason TEXT,
    latency_ms     REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id          TEXT NOT NULL REFERENCES audit_log(query_id),
    verdict           TEXT NOT NULL,
    correct_citations TEXT NOT NULL,
    reviewer          TEXT NOT NULL,
    comment           TEXT,
    created_at        TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


class _SqliteStore:
    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn


class SqliteAuditRepository(_SqliteStore):
    def record(self, entry: AuditRecord) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO audit_log VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    entry.query_id,
                    entry.timestamp,
                    entry.question,
                    entry.as_of.isoformat(),
                    json.dumps([{"chunk_id": h.chunk_id, "score": h.score} for h in entry.retrieved]),
                    entry.provider,
                    entry.prompt_version,
                    entry.status.value,
                    json.dumps([vars(c) for c in entry.citations]),
                    entry.refusal_reason,
                    entry.latency_ms,
                ),
            )

    def get(self, query_id: str) -> AuditRecord | None:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM audit_log WHERE query_id = ?", (query_id,)).fetchone()
        if row is None:
            return None
        return AuditRecord(
            query_id=row["query_id"],
            timestamp=row["timestamp"],
            question=row["question"],
            as_of=date.fromisoformat(row["as_of"]),
            retrieved=tuple(ScoredChunk(**h) for h in json.loads(row["retrieved"])),
            provider=row["provider"],
            prompt_version=row["prompt_version"],
            status=AnswerStatus(row["status"]),
            citations=tuple(Citation(**c) for c in json.loads(row["citations"])),
            latency_ms=row["latency_ms"],
            refusal_reason=row["refusal_reason"],
        )


class SqliteFeedbackRepository(_SqliteStore):
    def add(self, feedback: Feedback) -> int:
        with closing(self._connect()) as conn, conn:
            cursor = conn.execute(
                "INSERT INTO feedback (query_id, verdict, correct_citations, reviewer, comment) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    feedback.query_id,
                    feedback.verdict.value,
                    json.dumps(list(feedback.correct_citations)),
                    feedback.reviewer,
                    feedback.comment,
                ),
            )
            return int(cursor.lastrowid)

    def list_actionable(self) -> list[tuple[int, Feedback]]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM feedback WHERE verdict != ? ORDER BY id", (Verdict.CORRECT.value,)
            ).fetchall()
        return [
            (
                row["id"],
                Feedback(
                    query_id=row["query_id"],
                    verdict=Verdict(row["verdict"]),
                    reviewer=row["reviewer"],
                    correct_citations=tuple(json.loads(row["correct_citations"])),
                    comment=row["comment"],
                ),
            )
            for row in rows
        ]
