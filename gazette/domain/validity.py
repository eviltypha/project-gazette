"""As-of-date rule: which paragraphs are in force on a given date."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date

from gazette.domain.models import Circular, chunk_key


class ValidityCalendar:
    """A paragraph is in force on ``as_of`` when its circular is effective and no
    amendment replacing it has an effective date on or before ``as_of``."""

    def __init__(self, circulars: Iterable[Circular]) -> None:
        circulars = list(circulars)
        self._effective_from: dict[str, date] = {}
        self._superseded_from: dict[str, date] = {}
        for circular in circulars:
            for paragraph in circular.paragraphs:
                self._effective_from[paragraph.chunk_id] = circular.effective_date
        for circular in circulars:
            for amendment in circular.amends:
                for para in amendment.paras:
                    key = chunk_key(amendment.circular_id, para)
                    current = self._superseded_from.get(key)
                    if current is None or circular.effective_date < current:
                        self._superseded_from[key] = circular.effective_date

    def is_in_force(self, chunk_id: str, as_of: date) -> bool:
        effective = self._effective_from.get(chunk_id)
        if effective is None or effective > as_of:
            return False
        return not self.is_superseded(chunk_id, as_of)

    def is_superseded(self, chunk_id: str, as_of: date) -> bool:
        superseded = self._superseded_from.get(chunk_id)
        return superseded is not None and superseded <= as_of

    def in_force_ids(self, as_of: date) -> frozenset[str]:
        return frozenset(cid for cid in self._effective_from if self.is_in_force(cid, as_of))
