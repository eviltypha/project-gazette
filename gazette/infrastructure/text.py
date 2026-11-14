"""Deterministic text utilities shared by the keyword index, embedder, provider and judge."""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a an the of to in for and or on by with be is are was were been being shall should
    must may can could would will as at from that this these those which it its such any all
    under per than within into if not no so do does did what how when where who whom why
    often much many i we our us you your their there here also other""".split()
)

_TOKEN = re.compile(r"[a-z0-9]+")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.;:])\s+(?=[A-Z(])")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]
