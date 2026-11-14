"""Deterministic extractive provider used by tests and the default eval.

Each retrieved paragraph's best sentence is scored by the number of question terms
it shares, with a small bonus for stating a concrete figure (amount, period, count)
since officers' questions usually ask for one. The best-scoring sentence is cited,
plus a runner-up only if it scores as well, to keep citation precision high."""

from __future__ import annotations

import re
from collections.abc import Sequence

from gazette.application.ports import Draft
from gazette.domain.models import Citation, Paragraph
from gazette.infrastructure.text import split_sentences, tokenize

MAX_CITATIONS = 2
_FIGURE_BONUS = 0.5
_FIGURE = re.compile(r"(?i)\bRs\.?\s*\d|\b\d+\s*(?:per\s*cent|%|lakh|crore|working days?|days?|months?|years?)\b")


class FakeProvider:
    name = "fake-extractive"
    prompt_version = "extractive-v2"

    def generate(self, question: str, context: Sequence[Paragraph]) -> Draft:
        terms = set(tokenize(question))
        candidates = []
        for rank, paragraph in enumerate(context):
            sentence, score = _best_sentence(paragraph.text, terms)
            if score > 0:
                candidates.append((-score, rank, paragraph, sentence))
        candidates.sort(key=lambda c: (c[0], c[1]))
        picks = [c for c in candidates if c[0] == candidates[0][0]][:MAX_CITATIONS]
        return Draft(
            answer=" ".join(sentence for *_, sentence in picks),
            citations=tuple(Citation(p.circular_id, p.para, s) for _, _, p, s in picks),
        )


def _best_sentence(text: str, terms: set[str]) -> tuple[str, float]:
    best, best_score = "", 0.0
    for sentence in split_sentences(text):
        overlap = len(terms & set(tokenize(sentence)))
        if overlap == 0:
            continue
        score = overlap + (_FIGURE_BONUS if _FIGURE.search(sentence) else 0.0)
        if score > best_score:
            best, best_score = sentence, score
    return best, best_score
