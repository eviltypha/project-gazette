"""Deterministic text utilities shared by the keyword index, embedder, provider and judge."""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a an the of to in for and or on by with be is are was were been being shall should
    must may can could would will as at from that this these those which it its such any all
    under per than within into if not no so do does did what how when where who whom why
    often much many i we our us you your their there here also other""".split()
)

# Officers ask in acronyms; circulars spell terms out. Expanding both sides keeps
# BM25 and the hashed embedder consistent.
_PHRASES = (
    (re.compile(r"\bre-?kyc(?:'?d|ed)?\b"), "periodic updation kyc"),
    (re.compile(r"\bv-?cip\b"), "vcip video customer identification process"),
)
_ACRONYMS = {
    "str": "suspicious transaction report",
    "ctr": "cash transaction report",
    "pep": "politically exposed person",
    "fiu": "financial intelligence unit",
    "ubo": "beneficial owner",
    "edd": "enhanced due diligence",
    "cdd": "customer due diligence",
}

_TOKEN = re.compile(r"[a-z0-9]+")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.;:])\s+(?=[A-Z(])")


def _stem(token: str) -> str:
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    text = text.lower()
    for pattern, replacement in _PHRASES:
        text = pattern.sub(replacement, text)
    tokens: list[str] = []
    for raw in _TOKEN.findall(text):
        token = _stem(raw)
        expansion = _ACRONYMS.get(token)
        if expansion:
            tokens.extend(expansion.split())
        elif token not in STOPWORDS and len(token) > 1:
            tokens.append(token)
    return tokens


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]
