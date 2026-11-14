"""Command-line delivery: argument parsing and console output only."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from gazette.application.feedback import CandidateNotFoundError
from gazette.config import Settings
from gazette.container import approve_candidate, export_candidates, run_evaluation, run_ingest


def feedback_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m gazette.feedback",
                                     description="Turn reviewer feedback into eval candidates")
    parser.add_argument("--candidates", type=Path, default=Path("eval/candidates.jsonl"))
    parser.add_argument("--questions", type=Path, default=Path("eval/questions.jsonl"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("export", help="write non-correct feedback to the candidates file")
    approve = commands.add_parser("approve", help="move one candidate into the eval set")
    approve.add_argument("candidate_id")
    args = parser.parse_args(argv)

    if args.command == "export":
        items = export_candidates(Settings.from_env(), args.candidates)
        print(f"Exported {len(items)} candidate(s) to {args.candidates}")
        return 0
    try:
        item = approve_candidate(args.candidates, args.questions, args.candidate_id)
    except (CandidateNotFoundError, ValueError) as exc:
        print(f"error: cannot approve {args.candidate_id}: {exc}", file=sys.stderr)
        return 1
    print(f"Approved {item.id} into {args.questions}")
    return 0


def ingest_main(argv: Sequence[str] | None = None) -> int:
    argparse.ArgumentParser(prog="python -m gazette.ingest",
                            description="Build BM25 + vector indexes").parse_args(argv)
    settings = Settings.from_env()
    report = run_ingest(settings)
    print(
        f"Indexed {report.paragraphs} paragraphs from {report.circulars} circulars "
        f"({report.amending_circulars} amending) into {settings.index_dir}/"
    )
    return 0


def eval_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m gazette.eval",
                                     description="Run the labelled eval set")
    parser.add_argument("--questions", type=Path, default=Path("eval/questions.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("eval/report.json"))
    parser.add_argument("--min-recall", type=float, default=None,
                        help="exit non-zero if hybrid Recall@5 is below this value")
    args = parser.parse_args(argv)

    report = run_evaluation(Settings.from_env(), args.questions)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(format_report(report))
    print(f"\nWrote {args.report}")

    hybrid = report["recall_at_5"]["hybrid"]
    if args.min_recall is not None and hybrid < args.min_recall:
        print(f"FAIL: hybrid Recall@5 {hybrid:.4f} < --min-recall {args.min_recall:.4f}")
        return 1
    return 0


def format_report(report: dict) -> str:
    recall = report["recall_at_5"]
    sweep = report["threshold_sweep"]
    counts = report["counts"]
    lines = [
        f"Eval set: {counts['items']} items ({counts['answerable']} answerable, "
        f"{counts['should_refuse']} should-refuse, {counts['as_of']} as-of)",
        "",
        "| Retriever | Recall@5 |",
        "|---|---|",
        f"| BM25 only | {recall['bm25']:.4f} |",
        f"| Vector only | {recall['vector']:.4f} |",
        f"| Hybrid (RRF) | {recall['hybrid']:.4f} |",
        *([f"| Hybrid + cross-encoder rerank | {recall['reranked']:.4f} |"]
          if "reranked" in recall else []),
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Citation precision | {report['citation_precision']:.4f} |",
        f"| Refusal recall | {report['refusal_recall']:.4f} |",
        f"| False-refusal rate | {report['false_refusal_rate']:.4f} |",
        f"| As-of accuracy | {report['as_of_accuracy']:.4f} |",
        f"| Superseded / not-in-force citations | {report['superseded_citations']} |",
        f"| Groundedness | {report['groundedness']:.4f} |",
        f"| Refusal threshold (configured / selected by sweep) | "
        f"{report['config']['min_similarity']:.4f} / {sweep['selected']:.4f} |",
    ]
    return "\n".join(lines)
