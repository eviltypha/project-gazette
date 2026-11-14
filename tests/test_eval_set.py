"""The committed eval set must meet the assignment's size and composition rules."""

from datetime import date
from pathlib import Path

from gazette.domain.corpus import all_paragraphs
from gazette.domain.validity import ValidityCalendar
from gazette.infrastructure.json_corpus import JsonCorpusSource
from gazette.infrastructure.jsonl_eval_set import JsonlEvalSet
from tests.conftest import CORPUS_DIR

QUESTIONS = Path(__file__).resolve().parent.parent / "eval" / "questions.jsonl"
FAR_FUTURE = date(2100, 1, 1)


def test_eval_set_meets_minimum_counts():
    items = JsonlEvalSet(QUESTIONS).load()
    assert len(items) >= 40
    assert len({i.id for i in items}) == len(items)
    assert sum(not i.should_refuse for i in items) >= 25
    assert sum(i.should_refuse for i in items) >= 8
    assert sum(i.as_of is not None for i in items) >= 7


def test_gold_ids_exist_and_are_in_force_on_the_item_date():
    circulars = JsonCorpusSource(CORPUS_DIR).load()
    calendar = ValidityCalendar(circulars)
    known = {p.chunk_id for p in all_paragraphs(circulars)}
    for item in JsonlEvalSet(QUESTIONS).load():
        assert bool(item.gold) != item.should_refuse, item.id
        for chunk_id in item.gold:
            assert chunk_id in known, (item.id, chunk_id)
            assert calendar.is_in_force(chunk_id, item.as_of or date(2026, 11, 14)), item.id


def test_as_of_items_have_a_date_dependent_answer():
    """Each as-of item's gold sits in an amendment chain: it is either replaced later or
    belongs to an amending circular, so the correct citation depends on as_of."""
    circulars = JsonCorpusSource(CORPUS_DIR).load()
    calendar = ValidityCalendar(circulars)
    chain = {p.chunk_id for c in circulars if c.amends for p in c.paragraphs}
    chain |= {p.chunk_id for p in all_paragraphs(circulars) if calendar.is_superseded(p.chunk_id, FAR_FUTURE)}
    as_of_items = [i for i in JsonlEvalSet(QUESTIONS).load() if i.as_of]
    date_dependent = [i for i in as_of_items if set(i.gold) & chain]
    assert len(date_dependent) >= 7
    assert date_dependent == as_of_items
