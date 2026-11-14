"""End-to-end tests over the committed corpus; each run builds its own index in tmp_path."""

from collections import Counter
from dataclasses import replace
from datetime import date

import pytest
from fastapi.testclient import TestClient

from gazette.config import Settings
from gazette.container import run_ingest
from gazette.domain.corpus import all_paragraphs
from gazette.infrastructure.json_corpus import JsonCorpusSource
from gazette.interfaces.http import create_app
from tests.conftest import CORPUS_DIR


@pytest.fixture
def client():
    settings = Settings.from_env()
    run_ingest(settings)
    with TestClient(create_app(lambda: settings)) as test_client:
        yield test_client


def test_corpus_meets_minimum_counts():
    circulars = JsonCorpusSource(CORPUS_DIR).load()
    assert len(circulars) >= 8
    assert len(all_paragraphs(circulars)) >= 60
    assert sum(1 for c in circulars if c.amends) >= 3


def test_corpus_chunk_ids_are_unique_circular_para_keys():
    ids = [p.chunk_id for p in all_paragraphs(JsonCorpusSource(CORPUS_DIR).load())]
    assert not [i for i, n in Counter(ids).items() if n > 1]
    assert all(i.count("#") == 1 and i.split("#")[1].isdigit() for i in ids)


def test_known_question_is_answered_with_expected_citation(client):
    response = client.post(
        "/ask", json={"question": "Can V-CIP be used to onboard a PEP?", "as_of": "2024-01-01"}
    )

    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "answered"
    cited = {f"{c['circular_id']}#{c['para']}" for c in body["citations"]}
    assert "GZ-KYC-2023-02#3" in cited
    assert set(cited) <= set(body["retrieved"])


OFF_TOPIC_QUESTIONS = [
    "What is the current interest rate on home loans?",
    "What is the repo rate set by the central bank?",
    "How do I reset my net banking password?",
    "What is the minimum balance for a savings account?",
    "How do I open a demat account?",
    "What documents are needed for a car loan?",
    "Can I get a personal loan without a salary slip?",
    "What is the GST rate on bank charges?",
    "Who won the cricket match yesterday?",
]


@pytest.fixture(scope="module")
def module_client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("gazette")
    settings = replace(
        Settings.from_env(),
        corpus_dir=CORPUS_DIR,
        index_dir=tmp / "index",
        db_path=tmp / "gazette.sqlite3",
        llm_provider="fake",
    )
    run_ingest(settings)
    with TestClient(create_app(lambda: settings)) as test_client:
        yield test_client


@pytest.mark.parametrize("question", OFF_TOPIC_QUESTIONS)
def test_off_topic_question_is_refused(module_client, question):
    body = module_client.post("/ask", json={"question": question}).json()

    assert body["status"] == "refused"
    assert body["citations"] == []
    assert body["refusal_reason"]


def test_invalid_payload_is_rejected(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "ok?", "as_of": "not-a-date"}).status_code == 422


def test_default_as_of_is_today(client):
    body = client.post("/ask", json={"question": "What is the CTR threshold?"}).json()
    assert body["as_of"] == date.today().isoformat()
