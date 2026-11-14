"""As-of correctness on the real corpus: amended paragraphs on either side of the
amendment's effective date, and a circular that is not yet effective."""

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from gazette.config import Settings
from gazette.container import run_ingest
from gazette.interfaces.http import create_app
from tests.conftest import CORPUS_DIR


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("asof")
    settings = replace(Settings.from_env(), corpus_dir=CORPUS_DIR, index_dir=tmp / "index",
                       db_path=tmp / "gazette.sqlite3", llm_provider="fake")
    run_ingest(settings)
    with TestClient(create_app(lambda: settings)) as test_client:
        yield test_client


AMENDMENTS = [
    # question, last day before amendment, amendment effective date, old para, new para
    ("What is the CTR threshold?", "2021-09-30", "2021-10-01",
     "GZ-AML-2019-01#3", "GZ-AML-2021-02#3"),
    ("Within how many days must an STR be filed?", "2024-01-31", "2024-02-01",
     "GZ-AML-2019-01#5", "GZ-AML-2024-01#3"),
    ("How often must a high-risk customer be re-KYC'd?", "2023-03-31", "2023-04-01",
     "GZ-KYC-2019-02#4", "GZ-KYC-2023-01#3"),
]


def _ask(client, question, as_of):
    body = client.post("/ask", json={"question": question, "as_of": as_of}).json()
    cited = {f"{c['circular_id']}#{c['para']}" for c in body["citations"]}
    return body, cited


@pytest.mark.parametrize(("question", "before", "effective", "old", "new"), AMENDMENTS)
def test_original_paragraph_cited_before_amendment_takes_effect(client, question, before,
                                                                effective, old, new):
    body, cited = _ask(client, question, before)
    assert body["status"] == "answered"
    assert old in cited
    assert new not in body["retrieved"]


@pytest.mark.parametrize(("question", "before", "effective", "old", "new"), AMENDMENTS)
def test_replacement_paragraph_cited_from_amendment_effective_date(client, question, before,
                                                                   effective, old, new):
    body, cited = _ask(client, question, effective)
    assert body["status"] == "answered"
    assert new in cited
    assert old not in body["retrieved"]


def test_future_effective_circular_is_never_retrieved_before_its_effective_date(client):
    body, cited = _ask(client, "How long must transaction records be kept?", "2026-11-14")
    assert "GZ-AML-2020-01#3" in cited
    assert not any(cid.startswith("GZ-AML-2026-01") for cid in body["retrieved"])

    body, cited = _ask(client, "How long must transaction records be kept?", "2027-04-01")
    assert "GZ-AML-2026-01#3" in cited
    assert "GZ-AML-2020-01#3" not in body["retrieved"]
