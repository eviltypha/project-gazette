# Project Gazette

Citation-first Q&A service that answers compliance officers' KYC/AML questions **only** from
the synthetic RBI-style circulars in force on a given date, and refuses with
"not covered by the indexed circulars" when the circulars do not support an answer.

## What it does

- **Corpus**: 15 synthetic RBI-style circulars, 98 numbered paragraphs, 5 amending circulars
  (CTR threshold, high-risk re-KYC period, V-CIP for PEPs, STR timeline, and a record-retention
  change effective 2027-04-01, i.e. not yet in force). Near-duplicate paragraphs differ only in a
  number or date. All text is original; none is copied from real circulars.
- **`POST /ask`**: `{"question", "as_of"?}`. The response has `query_id`, `status`
  (`answered` or `refused`), `answer`, `citations` (`circular_id`, `para`, `quote`), `retrieved`,
  `as_of` and `refusal_reason`.
- **As-of correctness**: paragraphs that are superseded, or whose circular is not yet effective
  on `as_of` (default: today in IST), are filtered out before retrieval, so they can never be
  cited.
- **Citations checked in code**: each citation must point at a retrieved chunk and quote it
  verbatim. Failing citations are dropped; if none remain, the response is `refused`.
- **Audit and feedback**: every `/ask` is logged in SQLite (`GET /audit/{query_id}`), and
  officer feedback (`POST /feedback`) becomes eval candidates that a human approves.
- **Eval harness**: `python -m gazette.eval` reports retrieval, citation, refusal, as-of and
  groundedness metrics, and acts as a regression gate.

## Architecture

The package follows Clean Architecture. Dependencies point inwards only, and
`tests/test_architecture.py` fails the build if a layer imports an outer one.

```
 interfaces (FastAPI, CLIs) ──┐
                              ├──► application (use cases + ports) ──► domain (entities, rules)
 infrastructure (adapters) ───┘
            ▲
 container.py (composition root wires adapters into use cases)
```

| Layer | Package | Contents |
|---|---|---|
| Domain | `gazette/domain` | Entities (`Circular`, `Paragraph`, `Citation`, `Answer`, `Feedback`, `AuditRecord`, `EvalItem`) and pure rules: `ValidityCalendar` (as-of supersession), `validate_citations`, corpus integrity checks. No I/O, no frameworks. |
| Application | `gazette/application` | Use cases: `IngestCorpus`, `HybridRetriever` (RRF), `AskQuestion`, `AuditedAsk`, `EvaluateSystem`, `SubmitFeedback` / `ExportCandidates` / `ApproveCandidate`. Ports: `Embedder`, `SearchIndex`, `LLMProvider`, `Judge`, `AuditRepository`, `FeedbackRepository`, `EvalSetRepository`, `CorpusSource`. |
| Infrastructure | `gazette/infrastructure` | JSON corpus loader, BM25 index (`rank_bm25`), hashed TF-IDF embedder, NumPy cosine vector index, domain-aware tokeniser, extractive `FakeProvider`, `OverlapJudge`, SQLite audit and feedback repositories, JSONL eval store. |
| Interfaces | `gazette/interfaces` | FastAPI routes with Pydantic validation (`http.py`) and CLIs (`cli.py`). |
| Composition | `gazette/container.py` | Builds concrete adapters from `Settings` and injects them. |

### Request flow (`POST /ask`)

```
question, as_of
  └─► ValidityCalendar: keep paragraphs in force on as_of (drop superseded / not-yet-effective)
        └─► BM25 top-20  ∥  vector (cosine) top-20      (both restricted to in-force ids)
              └─► Reciprocal Rank Fusion → top-5
                    └─► refusal gate: best cosine < threshold → refused
                          └─► LLMProvider.generate(question, top-5 paragraphs)
                                └─► validate_citations (retrieved + verbatim) → none valid → refused
                                      └─► Judge: unsupported text → answer rebuilt from verified quotes
                                            └─► AuditedAsk writes the SQLite audit record
```

**Fusion: Reciprocal Rank Fusion.** `score(d) = Σ_r 1 / (60 + rank_r(d))` over the BM25 and
vector rankings. It uses ranks only, so BM25 and cosine scores never need normalising to a
common scale, and a paragraph found by only one retriever still scores. A test covers this.

**Retrievers.** BM25 rewards exact rare terms such as amounts, periods and acronyms. The vector
side is an offline, deterministic hashed word / bigram / character n-gram TF-IDF embedder
compared by cosine; it catches morphological variants that BM25 misses. Both share a tokeniser
that expands officer acronyms (STR, CTR, PEP, V-CIP, PO, re-KYC) and normalises plurals and
retention synonyms.

## Setup

Python 3.11+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Configuration is read from environment variables only; see `.env.example` for names and
defaults. The defaults run fully offline with the `FakeProvider` and the hashed embedder.

## Run

```bash
python -m gazette.ingest             # builds index/ (BM25 + vectors + manifest)
uvicorn gazette.api:app              # http://127.0.0.1:8000/docs

curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "What is the CTR threshold?", "as_of": "2020-06-01"}'
curl localhost:8000/audit/<query_id>
curl -X POST localhost:8000/feedback -H "Content-Type: application/json" \
     -d '{"query_id": "<query_id>", "verdict": "wrong_citation", "correct_citations": ["GZ-AML-2021-02#3"], "reviewer": "officer-1"}'
```

The service refuses to start if the corpus has changed since the last ingest, because the
manifest fingerprint no longer matches.

## Tests

```bash
pytest
```

The tests run offline, with no API keys, network or containers. Each test builds its own index
and SQLite database in a temp directory, so a committed `index/` is never needed. The `live`
marker is reserved for tests that would need a real provider; none exist yet.

## Evaluation

```bash
python -m gazette.eval                    # writes eval/report.json and prints the tables
python -m gazette.eval --min-recall 0.80  # exits 1 if hybrid Recall@5 < 0.80 (regression gate)
```

The eval builds a fresh index in a temp directory and runs `eval/questions.jsonl` with the
`FakeProvider`. That set has 52 officer-phrased questions: 42 answerable (11 of them as-of
questions across every amendment chain) and 10 plausible KYC/AML questions the corpus does not
cover (FATCA/CRS, crypto, correspondent banking, CKYCR and others). Questions without `as_of`
are evaluated at the fixed date 2026-11-14 so the report is reproducible.

### Results (fresh run, matches `eval/report.json`)

| Retriever | Recall@5 |
|---|---|
| BM25 only | 0.9524 |
| Vector only | 0.9286 |
| Hybrid (RRF) | 0.9524 |

| Metric | Value |
|---|---|
| Citation precision | 0.5918 |
| Refusal recall | 0.8000 |
| False-refusal rate | 0.1429 |
| As-of accuracy | 0.8182 |
| Superseded / not-in-force citations | 0 |
| Groundedness | 1.0000 |
| Refusal threshold (configured / selected by sweep) | 0.2319 / 0.2319 |

### How the refusal threshold is chosen

The refusal gate compares the best cosine similarity of the in-force vector hits against
`GAZETTE_MIN_SIMILARITY`. `python -m gazette.eval` records that similarity for every eval item
and sweeps every midpoint between consecutive observed values. Among thresholds whose
false-refusal rate on answerable items stays within the 0.15 operating budget, it picks the one
with the highest refusal recall on should-refuse items, breaking ties by the widest margin to
the nearest observation. The selected value (0.2319) is reported under `threshold_sweep` in
`report.json` and set as the default in `gazette/config.py`. If the two differ, the table above
shows it.

### Reading the numbers honestly

- **Hybrid equals BM25 here.** Fusion did not raise recall on this set. It is a safety net for
  phrasing where one retriever fails, and the vector-only row shows the embedder is still mostly
  lexical.
- **Groundedness is 1.0 by construction.** The `FakeProvider` is extractive. The `OverlapJudge`
  matters more for a real LLM, and the pipeline also uses it to strip unsupported claims.
- **Citation precision is 0.59.** Of the 20 non-gold citations, 7 are an extra second citation
  that tied with the correct one, 10 are the wrong paragraph chosen, often because topic terms
  such as "V-CIP" outweigh the attribute asked about, and 3 come from should-refuse questions
  that were answered.
- **The margin is thin.** The threshold sits 0.004 from the nearest observation, so unseen
  phrasing can land on either side.

## Feedback loop

- An officer reviews any answer with `POST /feedback`, giving a verdict, an optional
  `correct_citations` list and a reviewer name. Unknown `query_id`s return 404, and corrections
  citing paragraphs that are not in the corpus return 422.
- `python -m gazette.feedback export` turns every non-`correct` verdict into a candidate in
  `eval/candidates.jsonl`, in the same format as the eval set. The question and `as_of` come
  from the audit log, the gold comes from the reviewer's corrections, and
  `should_have_refused` becomes `should_refuse: true`.
- `python -m gazette.feedback approve <id>` moves exactly one candidate into
  `eval/questions.jsonl`. Nothing enters the regression set without that explicit step.
- **A reviewer can change:** gold citations and refusal expectations for future eval runs.
- **A reviewer cannot change:** the corpus, the audit record, live answers, retrieval or the
  threshold. Those change only through code or corpus edits plus a re-run of
  `python -m gazette.eval`.
- A `wrong_answer` with no corrected citations exports with an empty gold list. A human should
  fill it in before approving.

## Security and data handling

- No secrets in the repo. All configuration comes from environment variables, and real LLM
  providers are not wired in (off by default).
- Requests are strictly validated: `as_of` must be a `YYYY-MM-DD` string, unknown fields are
  rejected, and the question, comment and citation-list lengths are capped.
- The audit log stores question text verbatim. Officers may paste customer details, so in
  production this database should be access-controlled and retention-limited, and PII redaction
  should be considered before persisting.

## What I would do next

1. Replace the hashed embedder with a sentence-transformers model behind the same `Embedder`
   port to catch true paraphrases (e.g. "ambassador" as a PEP).
2. Add a cross-encoder reranker, or IDF-weighted sentence scoring in the provider, to lift
   citation precision. Cite a second paragraph only when it adds a distinct fact.
3. Contract acronyms to single tokens instead of expanding them, which keeps rare terms highly
   weighted in BM25.
4. Add a real `LLMProvider` (Azure OpenAI) with a versioned prompt, guarded by the same
   citation validation and judge.
5. Scope transition paragraphs to their own period, and make the judge negation-aware.
6. Add a provably safe answer cache keyed on corpus fingerprint, `as_of` and prompt version.

## Repository layout

```
gazette/          Python package (layers above); entry points ingest.py, api.py, eval.py, feedback.py
data/circulars/   Synthetic circular corpus (committed)
eval/             questions.jsonl (labelled set), report.json (latest run)
tests/            Offline pytest suite
index/            Generated by `python -m gazette.ingest` (git-ignored)
```
