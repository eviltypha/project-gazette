from datetime import date

from gazette.domain.models import Amendment, Circular, Paragraph
from gazette.domain.validity import ValidityCalendar


def _circular(cid, effective, paras, amends=()):
    return Circular(
        circular_id=cid,
        title=cid,
        issue_date=effective,
        effective_date=effective,
        paragraphs=tuple(Paragraph(cid, p, f"text {p}") for p in paras),
        amends=amends,
    )


CALENDAR = ValidityCalendar(
    [
        _circular("A", date(2024, 1, 1), ["1", "2"]),
        _circular("B", date(2025, 4, 1), ["1"], amends=(Amendment("A", ("2",)),)),
    ]
)


def test_amended_paragraph_in_force_until_amendment_effective():
    assert CALENDAR.is_in_force("A#2", date(2025, 3, 31))
    assert not CALENDAR.is_in_force("A#2", date(2025, 4, 1))
    assert CALENDAR.is_superseded("A#2", date(2025, 4, 1))


def test_circular_not_in_force_before_its_effective_date():
    assert not CALENDAR.is_in_force("B#1", date(2025, 3, 31))
    assert CALENDAR.is_in_force("B#1", date(2025, 4, 1))


def test_in_force_ids_on_date():
    assert CALENDAR.in_force_ids(date(2025, 6, 1)) == {"A#1", "B#1"}
    assert CALENDAR.in_force_ids(date(2023, 12, 31)) == set()
