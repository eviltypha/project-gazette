from datetime import date

from gazette.application.ask import REASON_NO_VALID_CITATION, AskQuestion
from gazette.application.ports import Draft
from gazette.application.retrieval import HybridRetriever
from gazette.domain.citations import validate_citations
from gazette.domain.models import AnswerStatus, Circular, Citation, Paragraph
from gazette.domain.validity import ValidityCalendar
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


class LyingProvider:
    name, prompt_version = "lying", "test"

    def generate(self, question, context):
        return Draft("CTRs are due in 3 days.", (Citation("GZ-A", "1", "CTRs are due in 3 days"),))


def test_pipeline_refuses_when_provider_citations_fail_validation():
    circular = Circular("GZ-A", "t", date(2020, 1, 1), date(2020, 1, 1), (P1, P2))
    ask = AskQuestion(
        retriever=HybridRetriever(StubIndex(["GZ-A#1"]), StubIndex(["GZ-A#1"])),
        provider=LyingProvider(),
        calendar=ValidityCalendar([circular]),
        paragraphs=PARAS,
        top_k=5,
        min_similarity=0.0,
    )

    answer = ask("When is the CTR due?", date(2024, 1, 1))

    assert answer.status is AnswerStatus.REFUSED
    assert answer.citations == ()
    assert answer.refusal_reason == REASON_NO_VALID_CITATION
