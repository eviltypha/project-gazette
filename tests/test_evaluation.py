import json

import pytest

from gazette.application.evaluation import sweep_threshold
from gazette.domain.models import Citation
from gazette.infrastructure.overlap_judge import OverlapJudge
from gazette.interfaces.cli import eval_main

QUOTE = Citation("GZ-A", "1", "The threshold for Cash Transaction Reports shall be Rs 8 lakh.")


def test_judge_scores_extractive_answer_as_fully_grounded():
    assert OverlapJudge().groundedness(QUOTE.quote, [QUOTE]) == 1.0


def test_judge_penalises_unsupported_sentence():
    answer = f"{QUOTE.quote} Branches may also waive the report for trusted customers."
    assert OverlapJudge().groundedness(answer, [QUOTE]) == 0.5


def test_threshold_sweep_picks_widest_gap_that_separates_classes():
    sweep = sweep_threshold(answerable=[0.30, 0.45, 0.60], refusable=[0.05, 0.10, 0.18])
    assert sweep["selected"] == pytest.approx(0.24)
    assert (sweep["refusal_recall"], sweep["false_refusal_rate"]) == (1.0, 0.0)


def _write(path, items):
    path.write_text("".join(json.dumps(i) + "\n" for i in items), encoding="utf-8")


def test_min_recall_gate_exits_non_zero_when_hybrid_recall_is_below(tmp_path):
    questions = tmp_path / "questions.jsonl"
    _write(questions, [
        {"id": "ok", "question": "What is the CTR threshold?", "gold": ["GZ-AML-2021-02#3"],
         "should_refuse": False},
        {"id": "miss", "question": "What is the CTR threshold?", "gold": ["GZ-KYC-2021-01#2"],
         "should_refuse": False},
    ])
    args = ["--questions", str(questions), "--report", str(tmp_path / "report.json")]

    assert eval_main([*args, "--min-recall", "0.5"]) == 0
    assert eval_main([*args, "--min-recall", "0.8"]) == 1
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["recall_at_5"]["hybrid"] == 0.5
