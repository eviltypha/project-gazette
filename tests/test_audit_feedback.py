from datetime import date

import pytest
from fastapi.testclient import TestClient

from gazette.application.feedback import to_candidate
from gazette.config import Settings
from gazette.container import run_ingest
from gazette.domain.models import AnswerStatus, AuditRecord, Feedback, Verdict
from gazette.infrastructure.jsonl_eval_set import JsonlEvalSet
from gazette.interfaces.cli import feedback_main
from gazette.interfaces.http import create_app

EXISTING = '{"id": "q001", "question": "Existing?", "gold": ["GZ-AML-2021-02#3"], "should_refuse": false}\n'


@pytest.fixture
def client():
    settings = Settings.from_env()
    run_ingest(settings)
    with TestClient(create_app(lambda: settings)) as test_client:
        yield test_client


def test_audit_record_for_refused_query_has_retrieved_ids_and_reason(client):
    body = client.post("/ask", json={"question": "What is the interest rate on home loans?"}).json()
    assert body["status"] == "refused"

    response = client.get(f"/audit/{body['query_id']}")

    record = response.json()
    assert response.status_code == 200
    assert record["status"] == "refused"
    assert record["retrieved"] and [h["chunk_id"] for h in record["retrieved"]] == body["retrieved"]
    assert all(isinstance(h["score"], float) for h in record["retrieved"])
    assert record["refusal_reason"] == body["refusal_reason"]
    assert record["question"] == "What is the interest rate on home loans?"
    assert record["provider"] == "fake-extractive" and record["prompt_version"]
    assert record["latency_ms"] >= 0 and record["as_of"] == body["as_of"]


def test_audit_record_for_answered_query_has_citations(client):
    body = client.post("/ask", json={"question": "What is the CTR threshold?"}).json()
    record = client.get(f"/audit/{body['query_id']}").json()
    assert record["status"] == "answered"
    assert record["citations"] == body["citations"]


def test_unknown_query_id_returns_404(client):
    for unknown in ("0" * 32, "does-not-exist"):
        assert client.get(f"/audit/{unknown}").status_code == 404
        response = client.post("/feedback", json={"query_id": unknown, "verdict": "correct",
                                                  "reviewer": "officer-1"})
        assert response.status_code == 404


def test_feedback_rejects_correction_citing_paragraph_not_in_corpus(client):
    query_id = client.post("/ask", json={"question": "What is the CTR threshold?"}).json()["query_id"]
    payload = {"query_id": query_id, "verdict": "wrong_citation", "reviewer": "r",
               "correct_citations": ["GZ-AML-2099-01#1"]}
    assert client.post("/feedback", json=payload).status_code == 422


def test_feedback_rejects_unknown_verdict_and_malformed_citation(client):
    query_id = client.post("/ask", json={"question": "What is the CTR threshold?"}).json()["query_id"]
    bad_verdict = {"query_id": query_id, "verdict": "meh", "reviewer": "r"}
    bad_citation = {"query_id": query_id, "verdict": "wrong_citation", "reviewer": "r",
                    "correct_citations": ["not a chunk id"]}
    assert client.post("/feedback", json=bad_verdict).status_code == 422
    assert client.post("/feedback", json=bad_citation).status_code == 422


def test_wrong_citation_feedback_reaches_eval_set_only_after_approval(client, tmp_path):
    questions, candidates = tmp_path / "questions.jsonl", tmp_path / "candidates.jsonl"
    questions.write_text(EXISTING, encoding="utf-8")
    question = "When is the monthly CTR due to FIU?"
    answered = client.post("/ask", json={"question": question, "as_of": "2025-01-01"}).json()
    correct = client.post("/ask", json={"question": "What is the CTR threshold?"}).json()

    for payload in (
        {"query_id": answered["query_id"], "verdict": "wrong_citation",
         "correct_citations": ["GZ-AML-2021-02#4"], "reviewer": "officer-1", "comment": "deadline"},
        {"query_id": correct["query_id"], "verdict": "correct", "reviewer": "officer-1"},
    ):
        assert client.post("/feedback", json=payload).status_code == 201

    paths = ["--candidates", str(candidates), "--questions", str(questions)]
    assert feedback_main([*paths, "export"]) == 0
    exported = JsonlEvalSet(candidates).load()
    assert len(exported) == 1
    assert [i.id for i in JsonlEvalSet(questions).load()] == ["q001"]

    assert feedback_main([*paths, "approve", exported[0].id]) == 0

    approved = JsonlEvalSet(questions).load()[-1]
    assert approved.id == exported[0].id
    assert approved.gold == ("GZ-AML-2021-02#4",)
    assert (approved.question, approved.as_of, approved.should_refuse) == (
        question, date(2025, 1, 1), False)
    assert JsonlEvalSet(candidates).load() == []
    assert feedback_main([*paths, "approve", exported[0].id]) == 1


def test_should_have_refused_becomes_should_refuse_candidate():
    record = AuditRecord("q", "t", "Is FATCA covered?", date(2026, 1, 1), (), "fake", "v",
                         AnswerStatus.ANSWERED, (), 1.0)
    feedback = Feedback("q", Verdict.SHOULD_HAVE_REFUSED, "officer-1")

    candidate = to_candidate(7, feedback, record)

    assert (candidate.id, candidate.should_refuse, candidate.gold) == ("fb-00007", True, ())
