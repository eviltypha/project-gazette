"""Corpus-level invariants that must hold before anything is indexed."""

from __future__ import annotations

from collections.abc import Sequence

from gazette.domain.models import Circular, Paragraph


class CorpusIntegrityError(ValueError):
    pass


def all_paragraphs(circulars: Sequence[Circular]) -> list[Paragraph]:
    return [p for c in circulars for p in c.paragraphs]


def check_integrity(circulars: Sequence[Circular]) -> None:
    """Unique circular ids and chunk ids; amendments target existing, earlier paragraphs."""
    by_id = {c.circular_id: c for c in circulars}
    if len(by_id) != len(circulars):
        raise CorpusIntegrityError("duplicate circular_id in corpus")
    chunk_ids = [p.chunk_id for p in all_paragraphs(circulars)]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise CorpusIntegrityError("duplicate paragraph number within a circular")
    known = set(chunk_ids)
    for circular in circulars:
        for amendment in circular.amends:
            target = by_id.get(amendment.circular_id)
            if target is None:
                raise CorpusIntegrityError(
                    f"{circular.circular_id} amends unknown circular {amendment.circular_id}"
                )
            if target.effective_date >= circular.effective_date:
                raise CorpusIntegrityError(
                    f"{circular.circular_id} must take effect after {target.circular_id}"
                )
            for para in amendment.paras:
                if f"{amendment.circular_id}#{para}" not in known:
                    raise CorpusIntegrityError(
                        f"{circular.circular_id} amends missing {amendment.circular_id}#{para}"
                    )
