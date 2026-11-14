# AI usage

## Tools

- **GitHub Copilot app (agent mode, Claude Opus 5.5)**: the main coding agent. It drafted the
  architecture, code, tests and docs, while I directed scope, reviewed the diffs and decided
  trade-offs.
- **Copilot sub-agents**, run in parallel:
  - *corpus generator*: wrote the 15 synthetic circulars and a validation script.
  - *eval-set author*: wrote and validated `eval/questions.jsonl` from the corpus, without
    running the system.
  - *stringent spec auditor* (rubber-duck reviewer): compared the spec with the code and
    listed gaps and traps.
  - *blind query generator*: produced held-out question sets knowing only the domain, with no
    access to the code or the spec.
  - a read-only Q&A chat, for my own questions while the main agent kept working.

## Key prompts (paraphrased)

1. "Perform initial analysis" of the assignment PDF, then "set up the project structure",
   then "ensure CLEAN architecture is followed with clear separation of concerns".
2. "Start a sub-agent to analyse RBI KYC/AML circulars and generate a synthetic dataset of
   about 15 circulars", constrained to: ≥8 circulars, ≥60 paragraphs, ≥3 amending circulars,
   and near-duplicates that differ **only** in a number or date.
3. "Check multiple off-topic questions, not just home loans"; "How about citation tests?";
   "Confidence on current task?"
4. Task 2 brief pasted verbatim; "parallelly start a sub-agent to look for gaps, traps and
   loopholes … make sure it is stringent".
5. "Test this: POST /feedback …" (a live HTTP test of the spec bullet).
6. "Generate 5 evaluation datasets … DO NOT provide it context related to the code or the
   requirements" (a blind, held-out evaluation).

## Where the AI was wrong and how it was caught

| # | What the AI got wrong | How it was caught | Fix |
|---|---|---|---|
| 1 | Refusal threshold 0.15 let off-topic questions (savings minimum balance, demat, personal loan) through as `answered` | I asked for a broader off-topic probe | Probed 18 off-topic, 8 uncovered and 12 covered questions; added filler/synonym normalisation; the threshold is now chosen from eval data |
| 2 | The extractive provider cited dated *transition* paragraphs for "current CTR threshold" | Clean-clone smoke test via `uvicorn` | Penalise sentences naming a calendar month; regression test added |
| 3 | `from __future__ import annotations` in `http.py` broke FastAPI's `Depends` in a closure, so every `/ask` returned 422 | Failing API tests | Removed the future import from the interface module |
| 4 | The first threshold sweep maximised Youden's J and picked a threshold with a 0.21 false-refusal rate, violating the ≤ 0.15 target | Stringent auditor sub-agent | Sweep now maximises refusal recall subject to the false-refusal budget |
| 5 | One genuine citation let an unsupported claim ride along in the answer text | Auditor's probe ("waive reporting for VIP customers") | Pipeline rebuilds the answer from verified quotes when the judge finds unsupported text; test added |
| 6 | A pattern check on `query_id` returned 422, not 404, for unknown non-hex ids | My live HTTP test of `/feedback` | Any unknown id now returns 404; test covers both shapes |
| 7 | Pydantic `strict=True` on `as_of` rejected valid ISO strings (FastAPI validates in Python mode), and non-strict accepted `0` as 1970-01-01 | Failing test, then the auditor | Custom validator that requires a `YYYY-MM-DD` string |
| 8 | Eval loader coerced `"false"` to `True` via `bool()` | Auditor | Explicit type validation |
| 9 | `.env.example` advertised an OpenAI provider that does not exist, and README said `cp .env.example .env` although nothing loads `.env` | Auditor | Removed the claim; documented env-only configuration |
| 10 | Created `config_new.py` / `text_new.py` side files instead of editing in place; a `git add --renormalize` folded unrelated files into one commit | My review of the file tree and `git show --stat` | Moved files into place; re-split the unpushed commits |
| 11 | The corpus agent could not express "V-CIP for PEPs prohibited → allowed" as a numbers-only near-duplicate | Its own validation report | Amendment changed to a credit-cap change; recorded as a judgement call |

## Verification performed in the session

- Every "Done means" bullet for Task 1 was checked against a fresh `git clone`: ingest,
  `uvicorn`, `/ask`, and `pytest` with no `index/` and network proxies pointed at a dead port.
- The eval numbers come from a fresh `python -m gazette.eval` run, and the README table is
  copied from its output.
- `POST /feedback` was exercised live over HTTP for all five verdicts, 404 and 422 cases, and
  the SQLite rows were inspected directly.
