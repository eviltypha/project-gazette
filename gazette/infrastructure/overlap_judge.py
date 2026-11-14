"""Deterministic groundedness judge: every answer sentence must be mostly made of
terms that appear in the cited quotes."""

from __future__ import annotations

from collections.abc import Sequence

from gazette.domain.models import Citation
from gazette.infrastructure.text import split_sentences, tokenize


class OverlapJudge:
    name = "token-overlap-v1"

    def __init__(self, min_overlap: float = 0.6) -> None:
        self._min_overlap = min_overlap

    def groundedness(self, answer: str, citations: Sequence[Citation]) -> float:
        sentences = split_sentences(answer)
        if not sentences or not citations:
            return 0.0
        quote_terms = {t for c in citations for t in tokenize(c.quote)}
        grounded = 0
        for sentence in sentences:
            terms = set(tokenize(sentence))
            if not terms or len(terms & quote_terms) / len(terms) >= self._min_overlap:
                grounded += 1
        return grounded / len(sentences)
