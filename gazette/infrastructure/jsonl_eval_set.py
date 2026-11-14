"""Eval set stored as JSON Lines (eval/questions.jsonl, eval/candidates.jsonl)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from gazette.domain.models import EvalItem


def item_to_dict(item: EvalItem) -> dict:
    payload: dict = {"id": item.id, "question": item.question}
    if item.as_of:
        payload["as_of"] = item.as_of.isoformat()
    payload["gold"] = list(item.gold)
    payload["should_refuse"] = item.should_refuse
    return payload


def item_from_dict(raw: dict) -> EvalItem:
    gold, should_refuse = raw.get("gold"), raw.get("should_refuse")
    if not isinstance(raw.get("id"), str) or not isinstance(raw.get("question"), str):
        raise ValueError(f"eval item needs string id and question: {raw!r}")
    if not isinstance(gold, list) or not all(isinstance(g, str) for g in gold):
        raise ValueError(f"eval item {raw['id']}: gold must be a list of chunk ids")
    if not isinstance(should_refuse, bool):
        raise ValueError(f"eval item {raw['id']}: should_refuse must be a boolean")
    return EvalItem(
        id=raw["id"],
        question=raw["question"],
        gold=tuple(gold),
        should_refuse=should_refuse,
        as_of=date.fromisoformat(raw["as_of"]) if raw.get("as_of") else None,
    )


class JsonlEvalSet:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def load(self) -> list[EvalItem]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [item_from_dict(json.loads(line)) for line in lines if line.strip()]

    def append(self, item: EvalItem) -> None:
        if any(existing.id == item.id for existing in self.load()):
            raise ValueError(f"eval item {item.id!r} already exists in {self.path}")
        self.write([*self.load(), item])

    def write(self, items: list[EvalItem]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        text = "".join(json.dumps(item_to_dict(i), ensure_ascii=False) + "\n" for i in items)
        self.path.write_text(text, encoding="utf-8", newline="\n")
