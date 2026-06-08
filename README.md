# ProdIntel AI

**An AI Product Decision Intelligence Platform** — it turns messy stakeholder signals
into **defensible, evidence-traceable product decisions**.

Not a chatbot. A decision-support system where every output can be walked back to the
exact stakeholder signals that produced it.

---

**Author:** Utkarsh Tiwari  
**Target roles:** AI Product Manager · Associate Product Manager · Product Strategy · Product Operations  
**Contact:** [Email](mailto:utkarsh7854@gmail.com) · [LinkedIn](https://www.linkedin.com/in/utkaxh/) · [GitHub](https://github.com/utkarshalpha)  
**Repository:** [github.com/utkarshalpha/prodintel-ai](https://github.com/utkarshalpha/prodintel-ai)  
**Status:** Stages 1–4 + Decision Explainability (`/why`) + framework-grounded decisioning (RAG) shipped · deployed Streamlit showcase (3 modes) · 569 tests (deterministic) · RICE *scoring engine* on the roadmap

![Python](https://img.shields.io/badge/python-3.11-blue)
![Tests](https://img.shields.io/badge/tests-569-brightgreen)
![Pydantic](https://img.shields.io/badge/pydantic-v2-e92063)
![FastAPI](https://img.shields.io/badge/FastAPI-009688)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0-d71f00)

---

## 🚀 Live Demo

**▶ [appintel-ai.streamlit.app](https://appintel-ai-d9uewikfgwfaktmlnzrfrh.streamlit.app/)** — deployed on Streamlit Community Cloud, no install or API key required.

The app has **three modes** (sidebar selector):

| Mode | What it shows |
|---|---|
| **Showcase** | A deterministic, narrated walkthrough of the full pipeline (signals → analysis → features → conflicts → decision → `/why` provenance) driven by a committed demo snapshot. The recruiter "happy path." |
| **Live Analysis** | Runs the **real pipeline** on *your* feedback (manual entry or CSV/TXT/PDF/DOCX upload). It uses a deterministic, input-derived **`LocalHeuristicClient`** — **no API key**, offline, and explicitly labeled *illustrative, not Claude-quality*: it exists to exercise the real validation gates, provenance, and `/why` on arbitrary input. |
| **Architecture** | The system's design rendered as Mermaid diagrams (with source). |

> The deployed demo makes **no Claude calls and needs no secret** — the Anthropic SDK is present only because the live pipeline imports the API package; real Claude execution remains a backend capability, not part of the public demo.

## ✨ Features

- **4-stage decision pipeline** — Signal Analysis → Feature Extraction → Conflict Detection → Decision Synthesis, each guarded by a **deterministic validation gate** that rejects any model output it cannot prove.
- **Decision Explainability** (`GET /decisions/{id}/why`) — walks the provenance graph from a decision back to the original stakeholder quotes, with a deterministic integrity check.
- **Framework-grounded decisioning (RAG)** — Stage 4 retrieves framework passages (RICE/Kano) from ChromaDB; the decision-integrity gate **rejects any citation not in the retrieved pool**, and citations persist as FK-protected edges.
- **Ingestion & validation** — manual entry + CSV/TXT/PDF/DOCX uploads → validated `FeedbackEntry` batches (size/row/length caps, partial-success reporting).
- **End-to-end orchestration** — a transaction-less `PipelineService` saga runs all stages and returns a canonical, partial-progress-aware result.
- **Deployed Streamlit showcase** — three modes, reusing the same renderers and snapshot assembler across the demo and the live path.

## 📸 Screenshots

> _Add PNGs to `docs/screenshots/` — they render below once present._

| Showcase | Live Analysis | Architecture |
|---|---|---|
| ![Showcase mode](docs/screenshots/showcase.png) | ![Live Analysis mode](docs/screenshots/live-analysis.png) | ![Architecture mode](docs/screenshots/architecture.png) |

---

> **Status:** Stages 1–4 of the decision pipeline are implemented, production-wired, and
> covered by **569 tests** — including **Decision Explainability** (`GET /decisions/{id}/why`),
> which walks a decision's provenance graph back to the original stakeholder signals, and
> **framework-grounded decisioning** (RAG: Stage 4 cites retrieved RICE/Kano passages, validated
> against the retrieved pool). A deterministic **RICE scoring engine** — rubric math that would
> replace the model-provided `priority_rank` — is still on the roadmap (see [§15](#15-roadmap)),
> and this README does not pretend otherwise.

---

## 1. Overview

ProdIntel AI ingests stakeholder signals (customer requests, sales asks, engineering
concerns, support tickets, leadership objectives), and runs them through a deterministic,
multi-stage AI pipeline. Each stage is a thin, typed wrapper over a shared runtime harness
and is guarded by a **deterministic validation gate** that rejects any model output it
cannot prove. Today the pipeline covers:

1. **Signal Analysis** — parse one signal into structured, *source-anchored* claims.
2. **Feature Extraction** — cluster grounded signals into normalized product features.
3. **Conflict Detection** — surface genuine disagreements between stakeholders.
4. **Decision Synthesis** — turn features, conflicts, and evidence into ranked,
   evidence-backed decisions that acknowledge every conflict over their subject.

The defining property is **traceability**: it is a data-model and validation invariant, not
a feature bolted on at the end.

## What This Project Demonstrates

A snapshot of the competencies this repository is meant to evidence — for AI-PM, product, and
engineering reviewers alike:

- **AI product thinking** — designed around a falsifiable trust boundary ("the model is
  untrusted; every output must be validated and traceable") rather than a thin LLM wrapper.
- **Product scoping judgment** — deliberately cut a Competitive-Intelligence engine and a
  standalone Roadmap engine to keep one *defensible* loop (signals → decisions). Breadth was
  traded for a provable core — a deliberate product decision, not an omission.
- **System design** — a layered, dependency-injected architecture (contracts → runtime harness →
  stages → services → API) with a single, swappable LLM boundary.
- **Trust & safety engineering** — deterministic validation gates (grounding, provenance, conflict
  integrity) that make unsupported or hallucinated output *impossible to persist*.
- **Reproducibility & eval-readiness** — deterministic retries and a faked LLM boundary so the full
  pipeline runs offline in tests; the foundation for a future No-RAG vs. RAG evaluation study.
- **Engineering discipline** — 569 deterministic tests, Alembic migrations with model-parity checks,
  structured JSON logging with correlation IDs, and hardened transaction boundaries.
- **Shipped product surface** — a deployed Streamlit app (three modes) with a real, no-API-key Live
  Analysis path over user uploads, an ingestion/validation layer, and a deterministic local engine —
  not just a backend API.
- **Documented decision-making** — 12 ADRs, each as *Problem → Decision → Trade-offs → Alternatives*
  (see [ARCHITECTURE_DECISIONS.md](ARCHITECTURE_DECISIONS.md)).

## Example End-to-End Flow

A walkthrough of the implemented pipeline. _Responses are illustrative and abbreviated; ids are
truncated. Requires the API running (see [§16](#16-running-locally)); the LLM boundary is faked in
tests and real in production._

```bash
# 1) Capture a stakeholder signal — immutable, deduplicated by content hash
curl -X POST localhost:8000/signals -H 'Content-Type: application/json' -d '{
  "source_type": "sales",
  "raw_text": "Enterprise SSO is critical for closing the Acme deal."
}'
# → 201  { "id": "8f3…", "source_type": "sales", "content_hash": "…", "created_at": "…" }
```

```bash
# 2) Analyze it (Stage 1) — every claim is anchored to a character span in raw_text (grounding)
curl -X POST localhost:8000/signals/8f3…/analyze
# → 200
# {
#   "analysis": {
#     "signal_id": "8f3…",
#     "intent": "Prioritize enterprise SSO to unblock a sales deal",
#     "stakeholder_type": "sales", "urgency": 5,
#     "claims": [
#       { "text": "Enterprise SSO is critical", "source_span": [0, 26], "claim_confidence": 0.95 }
#     ],
#     "confidence": { "score": 0.82, "basis": "strong", "components": { "span_grounding_ratio": 1.0 } }
#   },
#   "run": { "status": "success", "attempts_used": 1, "total_input_tokens": 1200, "total_output_tokens": 180 }
# }
```

```bash
# 3) Extract features (Stage 2) — each feature cites the signals it derives from (provenance)
curl -X POST localhost:8000/features/extract -d '{ "signal_ids": ["8f3…", "b21…"] }'
# → 200
# { "features": [ {
#     "id": "c47…", "title": "Enterprise SSO",
#     "jtbd": "When I evaluate vendors, I want SSO, so I can meet security policy",
#     "source_signal_ids": ["8f3…", "b21…"]
#   } ], "unassigned_signal_ids": [], "run": { … } }
```

```bash
# 4) Detect conflicts (Stage 3) — opposing positions, each evidence-backed
curl -X POST localhost:8000/conflicts/detect -d '{ "feature_ids": ["c47…"] }'
# → 200
# { "conflicts": [ {
#     "subject_id": "c47…", "conflict_type": "risk", "severity": 4,
#     "stakeholders": ["sales", "engineering"],
#     "positions": [
#       { "stakeholder": "sales", "stance": "advocate",
#         "summary": "SSO unblocks the Acme deal", "evidence_signal_ids": ["8f3…"] },
#       { "stakeholder": "engineering", "stance": "risk_flag",
#         "summary": "SSO is high-risk to build", "evidence_signal_ids": ["b21…"] }
#     ],
#     "evidence_signal_ids": ["8f3…", "b21…"],
#     "confidence": { "score": 0.78, "basis": "moderate" }
#   } ], "run": { … } }
```

```bash
# 5) Synthesize decisions (Stage 4) — a ranked, evidence-backed recommendation that
#    MUST acknowledge every conflict detected over its subject feature
curl -X POST localhost:8000/decisions/synthesize -d '{ "feature_ids": ["c47…"] }'
# → 200
# { "decisions": [ {
#     "id": "e90…", "subject_id": "c47…",
#     "recommendation": "build_now", "priority_rank": 1, "status": "proposed",
#     "title": "Build enterprise SSO",
#     "rationale": "Sales evidence supports it; the engineering risk conflict is acknowledged.",
#     "acknowledged_conflict_ids": ["d12…"],
#     "evidence_signal_ids": ["8f3…", "b21…"],
#     "confidence": { "score": 0.81, "basis": "strong" }
#   } ], "run": { … } }
```

**Follow the chain:** the decision's `evidence_signal_ids` point back to signals and its
`acknowledged_conflict_ids` back to the conflicts over its subject; each conflict's
`evidence_signal_ids` point back to signals; the feature's `source_signal_ids` point back to the
same signals; each claim points to a character span in the original `raw_text`. Nothing in the
pipeline exists without provenance — and any output the validators cannot substantiate (including a
decision that ignores a known conflict over its subject) is rejected, not persisted.

## 2. Why this project exists

Product Managers drown in conflicting input and synthesize it by hand. Generic AI tools will
happily produce a confident roadmap — but they can't show *why*, can't be audited, and will
hallucinate a "customer request" that nobody made. For a decision system, that's
disqualifying.

ProdIntel AI is built around the opposite premise: **the model is untrusted, and the system
must be able to prove every claim.** That single constraint drives the whole architecture —
contract-first I/O, deterministic validation gates, and a provenance graph from raw signal to
final artifact. It is also designed as a portfolio / research artifact, so runs are
**reproducible** (deterministic retries, no hidden randomness) and the engineering decisions
are documented as ADRs.

## 3. Product vision

**Core loop:**
`Stakeholder Signals → Signal Analysis → Feature Extraction → Conflict Detection →
Prioritization → Decision Recommendation → Evidence Traceability.`

**Primary users:** Senior / Group Product Managers and Product Operations teams.

**The thesis:** every recommendation must be traceable back to (a) the original stakeholder
signals, (b) the supporting evidence, (c) the frameworks applied, and (d) a confidence score.

The first four stages of this loop are implemented today, through decision synthesis — the
keystone — plus the **evidence-traceability endpoint** (`/why`) and **framework-grounded
decisioning** (RAG). A deterministic **RICE scoring engine** (rubric math replacing the
model-provided priority rank) is the next milestone.

## 4. Architecture diagram

Layered, dependency-injected, and contract-first. Everything off the LLM path is deterministic
and unit-tested.

```
┌──────────────────────────────────────────────────────────────────────────┐
│  HTTP  (FastAPI)   — routes + Pydantic schemas, correlation-id middleware  │
└───────────────────────────────┬──────────────────────────────────────────┘
                                 │
┌───────────────────────────────▼──────────────────────────────────────────┐
│  Services   — use cases, transaction boundaries (Read → AI → Write)        │
│  SignalService · FeatureService · ConflictService · DecisionService        │
└───────────────┬───────────────────────────────────┬──────────────────────┘
                │                                     │
   ┌────────────▼───────────┐            ┌────────────▼─────────────────────┐
   │  Repositories          │            │  AI Stages (one runner / stage)   │
   │  (SQLAlchemy, flush)   │            │  Stage1 · Stage2 · Stage3 · Stage4│
   └────────────┬───────────┘            └────────────┬─────────────────────┘
                │                                     │
   ┌────────────▼───────────┐            ┌────────────▼─────────────────────┐
   │  Postgres + Alembic    │            │  Runtime Harness                  │
   │  12 tables, provenance │            │  BaseStageRunner · RetryPolicy ·  │
   │  edges, indexes        │            │  StageResult · metrics            │
   └────────────────────────┘            └───────┬───────────────┬──────────┘
                                                 │               │
                                  ┌──────────────▼───┐  ┌────────▼──────────┐
                                  │ AI Contracts      │  │ Validators        │
                                  │ (Pydantic, strict)│  │ grounding ·       │
                                  └──────────────┬────┘  │ provenance ·      │
                                                 │       │ conflict integrity│
                                                 │       │ decision integrity│
                                  ┌──────────────▼────┐  └───────────────────┘
                                  │ Claude Adapter    │
                                  │ (ToolCallClient)  │  ← only LLM boundary
                                  └───────────────────┘

Cross-cutting:  Observability — structured JSON logging + correlation IDs
```

The Anthropic SDK sits behind a single `ToolCallClient` protocol, so the **entire system runs
in tests with no network and no API key.**

## 5. Stage 1: Signal Analysis

**Input:** one immutable stakeholder signal. **Output:** a structured analysis — intent,
stakeholder type, urgency, sentiment, and a set of **source-anchored claims**.

**Trust gate — Grounding.** Every extracted claim must carry a character span into the
original text, and a deterministic check verifies the span's tokens actually substantiate the
claim. Ungrounded claims fail and trigger a retry; a parse with zero grounded claims is
rejected. **The system cannot persist a claim the source text does not support.**

> _Example:_ a claim *"users cannot request a refund"* pointing its span at the text
> *"peak hours"* is rejected deterministically — overlap below threshold.

## 6. Stage 2: Feature Extraction

**Input:** multiple grounded analyses. **Output:** normalized product features, each with a
Jobs-to-be-Done statement and the signals it derives from.

**Trust gate — Provenance.** Every feature's `source_signal_ids` must be real input signals,
and the inputs must be **fully partitioned** — each signal appears in exactly one feature or in
an explicit `unassigned` list, never both, never neither. Provenance is also persisted
relationally (`feature_signal` edge table with a foreign key that **prevents deleting a signal
that backs a feature**).

## 7. Stage 3: Conflict Detection

**Input:** features and the stakeholder signals behind them. **Output:** detected conflicts —
classified as priority / risk / resource / strategic — each with per-stakeholder positions and
the evidence backing them.

**Trust gate — Conflict integrity.** A conflict must reference a real subject feature, cite real
evidence signals for each position, and contain **genuine opposition** (≥2 distinct non-neutral
stances). An **empty result is valid** — it asserts the stakeholders agree — which stops the
model from manufacturing drama to seem useful.

> _Example:_ Sales says *"Enterprise SSO is critical"*, Engineering flags *"SSO is high risk"* →
> a `risk` conflict with advocate vs. risk-flag positions, each traceable to its signal.

## 8. Stage 4: Decision Synthesis

**Input:** features, the stakeholder signals behind them, and the conflicts already detected
over them. **Output:** the keystone artifact — a set of ranked, **evidence-backed decisions**,
each with a recommendation (`build_now` / `build_later` / `reject` / `needs_discussion`), a
rationale, a priority rank, the signals it rests on, and the conflicts it accounts for.

**Trust gate — Decision integrity.** A decision must be about a real input feature, cite only
real input signals as evidence, and acknowledge only real conflicts that are about *its own*
subject. Most importantly, it must acknowledge **every** conflict detected over that subject —
a decision **cannot silently ignore a known conflict** about the feature it decides. No two
decisions may decide the same feature. Any violation fails the stage and triggers a retry, so a
persisted decision provably accounts for everything known to oppose it (see ADR-012).

Traceability is enforced relationally, not in prose: a decision's evidence and its acknowledged
conflicts are real edge tables (`decision_evidence`, `decision_conflict`) with foreign keys that
**prevent deleting a signal or conflict a decision depends on** — the same deletion protection
`feature_signal` gives features.

> _Example:_ a "build now" decision on **Enterprise SSO** that cites the Sales signal but never
> acknowledges the Engineering risk conflict detected over that very feature is rejected
> deterministically — a decision may not look past a conflict it already knows about.

## 9. Trust & Validation Architecture

Two tiers per stage, both deterministic:

- **Schema validation** (Pydantic) — strict contracts (`extra="forbid"`, `frozen=True`); a model
  that invents a field fails loudly instead of corrupting state. The tool schema the model sees
  and the validator the system runs are generated from the *same* contract.
- **Semantic validation** (stage-specific gate) — pure, dependency-free functions:
  `grounding` (Stage 1), `provenance` (Stage 2), `conflict_integrity` (Stage 3),
  `decision_integrity` (Stage 4).

When a gate fails, it produces **deterministic retry feedback** (the exact violated constraint +
a hint), so the same failure always yields the same correction — replayable for research. The
harness is the *only* retry authority (capped exponential backoff, no jitter; the SDK's internal
retries are disabled).

See [ARCHITECTURE_DECISIONS.md](ARCHITECTURE_DECISIONS.md) for the full rationale (12 ADRs).

## 10. Tech Stack

| Concern | Choice |
|---|---|
| Language | Python 3.11 |
| API | FastAPI |
| Data model / ORM | SQLAlchemy 2.0 |
| Migrations | Alembic |
| Validation / contracts | Pydantic v2 |
| LLM | Anthropic Claude (via a `ToolCallClient` adapter) |
| Database | PostgreSQL (production); SQLite for the dev/test default |
| Vector store / RAG | ChromaDB (framework corpus) + an in-memory adapter for tests/demo |
| Ingestion | stdlib CSV/TXT · `pypdf` (PDF) · `python-docx` (DOCX) |
| UI / showcase | Streamlit (deployed on Streamlit Community Cloud) |
| Testing | pytest |
| Observability | stdlib `logging` with a JSON formatter + correlation IDs |

## 11. API Overview

Entry point: `uvicorn app.main:app`. Every request carries a correlation id (`X-Request-ID`,
generated if absent).

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/signals` | Create a signal (idempotent by content hash → 201 new / 200 existing) |
| `GET` | `/signals/{id}` | Retrieve a signal |
| `POST` | `/signals/{id}/analyze` | Run Stage 1; persist grounded analysis |
| `GET` | `/signals/{id}/analysis` | Retrieve the persisted analysis |
| `POST` | `/features/extract` | Run Stage 2 over analyzed signals |
| `GET` | `/features/{id}` | Retrieve a feature (with its source signals) |
| `GET` | `/features` | List features (`?signal_id`, `?workspace_id`) |
| `POST` | `/conflicts/detect` | Run Stage 3 over features |
| `GET` | `/conflicts/{id}` | Retrieve a conflict (with positions + evidence) |
| `GET` | `/conflicts` | List conflicts (`?subject_id`, `?workspace_id`) |
| `POST` | `/decisions/synthesize` | Run Stage 4 over features (and their conflicts) |
| `GET` | `/decisions/{id}` | Retrieve a decision (with evidence + acknowledged conflicts) |
| `GET` | `/decisions/{id}/why` | Explain a decision — walk its provenance graph back to original signals |
| `GET` | `/decisions` | List decisions (`?subject_id`, `?workspace_id`) |

Stage failures return a structured `422` (status, attempts, error code). Stage runners are read
from app state and return `503` if not configured.

## 12. Database Schema

Twelve tables. UUID primary keys; enums stored as lowercase values; JSON columns for structured AI
output. Framework knowledge is embedded into ChromaDB and cited by decisions; `signal.chroma_id`
remains reserved for future signal-level RAG.

| Table | Purpose | Key integrity |
|---|---|---|
| `signal` | Immutable stakeholder signal (provenance root) | `UNIQUE(content_hash)`; ORM-level immutability guard |
| `parsed_signal` | Grounded Stage 1 analysis (1:1) | `UNIQUE(signal_id)`, FK→signal `CASCADE` |
| `feature` | Normalized product feature | — |
| `feature_signal` | Provenance edge feature↔signal | composite PK; FK→signal **`RESTRICT`**; reverse-lookup index |
| `conflict` | Detected disagreement over a subject | polymorphic `subject_id` (indexed) |
| `conflict_party` | One stakeholder position + evidence | FK→conflict `CASCADE` |
| `decision` | Synthesized, evidence-backed decision over a subject | polymorphic `subject_id` (indexed); `recommendation` / `priority_rank` / `status` |
| `decision_evidence` | Provenance edge decision↔signal | composite PK; FK→signal **`RESTRICT`**; reverse-lookup index |
| `decision_conflict` | Edge decision↔acknowledged conflict | composite PK; FK→conflict **`RESTRICT`**; reverse-lookup index |
| `knowledge_source` | A framework document in the RAG corpus (RICE, Kano, …) | `framework` / `title` / `corpus_version` |
| `knowledge_chunk` | An embedded, vector-indexed passage of a source | FK→knowledge_source `CASCADE`; `chroma_id` |
| `decision_framework_citation` | Edge decision↔cited framework chunk | composite PK; FK→knowledge_chunk **`RESTRICT`**; `retrieval_score` / `relationship_type` / `evidence_type` |

Stage 4 makes a decision's evidence and its acknowledged conflicts **FK-protected edge tables**
(deletion protection), not JSON — so neither a backing signal nor an acknowledged conflict can be
deleted out from under a decision.

Current Alembic head: **`0007_create_decision_framework_citation`** (migrations 0001–0007).
Migration↔model parity (columns, types, constraints, foreign keys, indexes) is enforced by a test
across all twelve tables.

`workspace_id` (all tables) and `chroma_id` (`signal`) are **reserved/nullable** for upcoming
workspace-scoping and signal-level RAG — present in the schema, not yet used.

## 13. Testing

**569 tests** (565 passing + 4 environment-gated `chromadb` integration tests that skip unless
`RUN_CHROMA_TESTS=1`), fully deterministic — no network, no API key. The LLM boundary is faked
behind the `ToolCallClient` protocol, so the full pipeline (including retries and validation) runs
in-process in seconds.

```bash
python -m pytest          # full suite
python -m pytest -v       # verbose
```

Coverage spans contracts, each deterministic gate (grounding, provenance, conflict integrity,
decision integrity — with adversarial cases), the runtime harness, the Claude adapter (retryable
vs. fatal error mapping), services (Stage 1→2→3→4 end-to-end), the HTTP API, migrations
(apply/rollback + parity), provenance integrity (deletion protection + reverse traversal),
framework-grounded retrieval and citation persistence, the ingestion layer (CSV/TXT/PDF/DOCX +
validation), the local heuristic engine, the snapshot assembler (byte-identical regression), and
structured logging.

## 14. Documentation

- **[PROJECT_STATE.md](PROJECT_STATE.md)** — living snapshot: architecture, completed stages,
  schema, endpoints, runtime/validation/observability, test count, Alembic head, open tech debt,
  and roadmap.
- **[ARCHITECTURE_DECISIONS.md](ARCHITECTURE_DECISIONS.md)** — 12 ADRs, each as
  *Problem → Decision → Trade-offs → Alternatives considered* (contract-first, harness-managed
  metadata, deterministic retries, the four validation gates, repository & service patterns,
  structured logging, transaction boundaries, evidence traceability, the decision-integrity and
  conflict-acknowledgment invariant).

## 15. Roadmap

Implemented today: **Stages 1–4** + runtime harness, Claude adapter, structured logging, Alembic
migrations, hardened transaction boundaries. **Decision Synthesis (Stage 4) — the keystone — is
shipped:** ranked, evidence-backed decisions that must acknowledge every conflict over their
subject (the conflict-acknowledgment invariant, ADR-012). **Decision Explainability (Phase 5) is
shipped:** `GET /decisions/{id}/why` walks the provenance graph (decision → evidence /
acknowledged-conflict edges → subject feature → conflicts → signals → claims/spans) back to the
original stakeholder text, with a deterministic integrity check and `quoted_text` for every claim.

**Also shipped since:** **Framework-grounded decisioning (RAG)** — ChromaDB + a framework corpus,
Stage 4 citing retrieved passages with citations validated against the retrieved pool · the
`PipelineService` orchestrator · the **ingestion & validation** layer (CSV/TXT/PDF/DOCX) · the
deployed **Streamlit showcase** (three modes).

**Next (P0 — finish the decision loop):**
- **RICE scoring engine** — deterministic priority math in code that *replaces* the model-provided
  `priority_rank` with a rubric (distinct from the already-shipped RICE/Kano *framework citations*).
- **Confidence service** — computed Evidence Coverage / Reasoning Quality / Input Confidence.

**P1:** Workspace entity & scoping · Decision history / audit log · Eval harness (No-RAG vs RAG).

**Out of scope:** auth, billing, notifications, real tool connectors, competitive intelligence.

*(Items under "Next" and "P1" are not implemented yet; those sections describe intended work, not
shipped features.)*

## 16. Running Locally

**Prerequisites:** Python 3.11+, and (for production mode) PostgreSQL.

```bash
# 1) Install dependencies (the project runs from the repo root — no build/install step)
python -m pip install -r requirements.txt
python -m pip install pytest httpx              # test-only deps

# 2) Run the test suite — no API key or database required
python -m pytest
```

**Run the API against Claude + Postgres:**

```bash
# Configure the model + database via environment variables
export ANTHROPIC_API_KEY="sk-ant-..."            # required to serve requests
export ANTHROPIC_MODEL="claude-sonnet-4-6"       # optional (sensible default)
export DATABASE_URL="postgresql+psycopg://user:pass@localhost/prodintel"

# Apply migrations (Postgres needs a driver, e.g. pip install "psycopg[binary]")
alembic upgrade head

# Serve
uvicorn app.main:app
```

The dev/test default uses SQLite, so the test suite and local experimentation need neither a
database server nor an API key. A real `ANTHROPIC_API_KEY` is required only to serve live requests
through the running API.

**Run the Streamlit showcase (no API key, no database):**

```bash
python -m pip install -r requirements.txt
streamlit run showcase/app.py
```

This launches the three-mode app (Showcase / Live Analysis / Architecture) locally. Live Analysis
uses the deterministic `LocalHeuristicClient`, so it runs fully offline — no Anthropic key.

**Deploy to Streamlit Community Cloud:**

1. Push the repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), create an app pointing at this repo, with
   **Main file path = `showcase/app.py`**.
3. Cloud installs from `requirements.txt` automatically. **No secrets are required** — the deployed
   demo uses the local engine and makes no Claude calls.

> Note on dependencies: `requirements.txt` (repo root) and `showcase/requirements.txt` (next to the
> entrypoint) are kept identical; Streamlit Cloud resolves the one adjacent to the entrypoint.

---

_Built with a strict separation between what the model produces and what the system will trust.
Every claim, feature, and conflict in ProdIntel AI is traceable to the stakeholder signal it came
from._
