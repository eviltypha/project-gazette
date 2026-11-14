"""Citation validation: model citations are never trusted, they are checked here."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from gazette.domain.models import Citation, Paragraph

MIN_QUOTE_CHARS = 20


@dataclass(frozen=True)
class RejectedCitation:
    citation: Citation
    reason: str


@dataclass(frozen=True)
class CitationCheck:
    valid: tuple[Citation, ...]
    rejected: tuple[RejectedCitation, ...]


def validate_citations(
    citations: Sequence[Citation],
    retrieved_ids: Collection[str],
    paragraphs: Mapping[str, Paragraph],
) -> CitationCheck:
    """Keep a citation only if it points at a retrieved chunk and its quote is a
    verbatim substring (of meaningful length) of that paragraph's text.

    Retrieval is already restricted to paragraphs in force on the as-of date, so
    "retrieved" also implies "citable on that date"."""
    valid: list[Citation] = []
    rejected: list[RejectedCitation] = []
    seen: set[tuple[str, str]] = set()
    for citation in citations:
        reason = _rejection_reason(citation, retrieved_ids, paragraphs)
        if reason:
            rejected.append(RejectedCitation(citation, reason))
        elif (citation.chunk_id, citation.quote) not in seen:
            seen.add((citation.chunk_id, citation.quote))
            valid.append(citation)
    return CitationCheck(tuple(valid), tuple(rejected))


def _rejection_reason(
    citation: Citation, retrieved_ids: Collection[str], paragraphs: Mapping[str, Paragraph]
) -> str | None:
    if citation.chunk_id not in retrieved_ids:
        return "cited paragraph was not retrieved"
    if len(citation.quote.strip()) < MIN_QUOTE_CHARS:
        return "quote too short to verify"
    if citation.quote not in paragraphs[citation.chunk_id].text:
        return "quote is not a verbatim substring of the paragraph"
    return None
