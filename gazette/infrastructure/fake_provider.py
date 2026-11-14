"""Deterministic extractive provider used by tests and the default eval.

For each context paragraph (in retrieval order) it picks the sentence sharing the
most terms with the question. It always cites the top paragraph and adds the next
ones only when they match the question at least as well, to keep precision high."""

from __future__ import annotations

from collections.abc import Sequence

from gazette.application.ports import Draft
from gazette.domain.models import Citation, Paragraph
from gazette.infrastructure.text import split_sentences, tokenize

MAX_CITATIONS = 2


class FakeProvider:
    name = "fake-extractive"
    prompt_version = "extractive-v1"

    def generate(self, question: str, context: Sequence[Paragraph]) -> Draft:
        terms = set(tokenize(question))
        picks: list[tuple[Paragraph, str, int]] = []
        for paragraph in context:
            sentence, overlap = _best_sentence(paragraph.text, terms)
            if overlap == 0 or (picks and overlap < picks[0][2]):
                continue
            picks.append((paragraph, sentence, overlap))
            if len(picks) == MAX_CITATIONS:
                break
        return Draft(
            answer=" ".join(sentence for _, sentence, _ in picks),
            citations=tuple(Citation(p.circular_id, p.para, s) for p, s, _ in picks),
        )


def _best_sentence(text: str, terms: set[str]) -> tuple[str, int]:
    best, best_overlap = "", 0
    for sentence in split_sentences(text):
        overlap = len(terms & set(tokenize(sentence)))
        if overlap > best_overlap:
            best, best_overlap = sentence, overlap
    return best, best_overlap
