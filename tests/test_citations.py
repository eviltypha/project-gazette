from datetime import date

from gazette.application.ask import REASON_NO_VALID_CITATION, AskQuestion
from gazette.application.ports import Draft
from gazette.application.retrieval import HybridRetriever
from gazette.domain.citations import validate_citations
from gazette.domain.models import Amendment, AnswerStatus, Circular, Citation, Paragraph
from gazette.domain.validity import ValidityCalendar
from gazette.infrastructure.overlap_judge import OverlapJudge
from tests.helpers import StubIndex

P1 = Paragraph("GZ-A", "1", "Cash Transaction Reports shall be filed by the 15th of the next month.")
P2 = Paragraph("GZ-A", "2", "Suspicious Transaction Reports shall be filed within seven working days.")
PARAS = {p.chunk_id: p for p in (P1, P2)}


def test_valid_verbatim_citation_is_kept():
    citation = Citation("GZ-A", "1", "filed by the 15th of the next month")
    check = validate_citations([citation], {"GZ-A#1"}, PARAS)
    assert check.valid == (citation,)


def test_non_verbatim_quote_is_dropped():
    paraphrase = Citation("GZ-A", "1", "filed by the fifteenth of the following month")
    check = validate_citations([paraphrase], {"GZ-A#1"}, PARAS)
    assert check.valid == ()
    assert check.rejected[0].reason == "quote is not a verbatim substring of the paragraph"


def test_citation_to_unretrieved_paragraph_is_dropped():
    fabricated = Citation("GZ-A", "2", "filed within seven working days")
    check = validate_citations([fabricated], {"GZ-A#1"}, PARAS)
    assert check.valid == ()
    assert check.rejected[0].reason == "cited paragraph was not retrieved"


def test_verbatim_quote_attributed_to_wrong_paragraph_is_dropped():
    misattributed = Citation("GZ-A", "1", "filed within seven working days")
    check = validate_citations([misattributed], {"GZ-A#1", "GZ-A#2"}, PARAS)
    assert check.valid == ()


def test_case_altered_quote_is_not_verbatim():
    altered = Citation("GZ-A", "1", "cash transaction reports shall be filed")
    assert validate_citations([altered], {"GZ-A#1"}, PARAS).valid == ()


def test_too_short_quote_is_dropped_even_if_verbatim():
    check = validate_citations([Citation("GZ-A", "1", "shall be")], {"GZ-A#1"}, PARAS)
    assert check.rejected[0].reason == "quote too short to verify"


def test_mixed_citations_keep_only_valid_and_deduplicate():
    good = Citation("GZ-A", "1", "filed by the 15th of the next month")
    bad = Citation("GZ-A", "1", "filed by the 20th of the next month")
    check = validate_citations([good, bad, good], {"GZ-A#1"}, PARAS)
    assert check.valid == (good,)
    assert [r.citation for r in check.rejected] == [bad]


class LyingProvider:
    name, prompt_version = "lying", "test"

    def generate(self, question, context):
        return Draft("CTRs are due in 3 days.", (Citation("GZ-A", "1", "CTRs are due in 3 days"),))


def test_pipeline_refuses_when_provider_citations_fail_validation():
    circular = Circular("GZ-A", "t", date(2020, 1, 1), date(2020, 1, 1), (P1, P2))
    ask = AskQuestion(
        retriever=HybridRetriever(StubIndex(["GZ-A#1"]), StubIndex(["GZ-A#1"])),
        provider=LyingProvider(),
        judge=OverlapJudge(),
        calendar=ValidityCalendar([circular]),
        paragraphs=PARAS,
        top_k=5,
        min_similarity=0.0,
    )

    answer = ask("When is the CTR due?", date(2024, 1, 1))

    assert answer.status is AnswerStatus.REFUSED
    assert answer.citations == ()
    assert answer.refusal_reason == REASON_NO_VALID_CITATION


OLD = Paragraph("GZ-OLD", "1", "The threshold for Cash Transaction Reports shall be Rs 10 lakh.")
NEW = Paragraph("GZ-NEW", "1", "The threshold for Cash Transaction Reports shall be Rs 8 lakh.")


class CitesEverythingProvider:
    """Cites every paragraph it can see plus the superseded one it 'remembers'."""

    name, prompt_version = "cites-everything", "test"

    def generate(self, question, context):
        cited = [*context, OLD]
        return Draft("answer", tuple(Citation(p.circular_id, p.para, p.text) for p in cited))


def test_pipeline_drops_citation_of_superseded_paragraph():
    calendar = ValidityCalendar([
        Circular("GZ-OLD", "t", date(2019, 1, 1), date(2019, 1, 1), (OLD,)),
        Circular("GZ-NEW", "t", date(2021, 1, 1), date(2021, 10, 1), (NEW,),
                 amends=(Amendment("GZ-OLD", ("1",)),)),
    ])
    ask = AskQuestion(
        retriever=HybridRetriever(StubIndex(["GZ-OLD#1", "GZ-NEW#1"]), StubIndex(["GZ-OLD#1"])),
        provider=CitesEverythingProvider(),
        judge=OverlapJudge(),
        calendar=calendar,
        paragraphs={p.chunk_id: p for p in (OLD, NEW)},
        top_k=5,
        min_similarity=0.0,
    )

    answer = ask("CTR threshold?", date(2022, 1, 1))

    assert answer.status is AnswerStatus.ANSWERED
    assert [c.chunk_id for c in answer.citations] == ["GZ-NEW#1"]
    assert "GZ-OLD#1" not in [h.chunk_id for h in answer.retrieved]


class SmuggledClaimProvider:
    """Pairs a genuine quote with an unsupported extra claim."""

    name, prompt_version = "smuggler", "test"

    def generate(self, question, context):
        quote = "Cash Transaction Reports shall be filed by the 15th of the next month."
        return Draft(f"{quote} Branches may waive reporting for VIP customers.",
                     (Citation("GZ-A", "1", quote),))


def test_pipeline_strips_unsupported_claims_riding_on_a_valid_citation():
    circular = Circular("GZ-A", "t", date(2020, 1, 1), date(2020, 1, 1), (P1, P2))
    ask = AskQuestion(
        retriever=HybridRetriever(StubIndex(["GZ-A#1"]), StubIndex(["GZ-A#1"])),
        provider=SmuggledClaimProvider(),
        judge=OverlapJudge(),
        calendar=ValidityCalendar([circular]),
        paragraphs=PARAS,
        top_k=5,
        min_similarity=0.0,
    )

    answer = ask("When is the CTR due?", date(2024, 1, 1))

    assert answer.status is AnswerStatus.ANSWERED
    assert answer.answer == P1.text
    assert "VIP" not in answer.answer
