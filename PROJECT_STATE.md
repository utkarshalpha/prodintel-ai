# ProdIntel AI — Project State

_Last updated: 2026-06-05_

A living snapshot of the system: what it does, how it is built, what is done, and what
remains. Companion to [ARCHITECTURE_DECISIONS.md](ARCHITECTURE_DECISIONS.md), which
records _why_ the major choices were made.

---

## 1. Product Vision

ProdIntel AI is an **AI Product Decision Intelligence Platform**. It transforms messy
stakeholder signals into **defensible, evidence-traceable product decisions**. It is
not a chatbot; it is a decision-support system for Senior/Group Product Managers and
Product Operations teams.

**Core loop:** `Stakeholder Signals → Signal Analysis → Feature Extraction → Conflict
Detection → Prioritization → Decision Recommendation → Evidence Traceability.`

**The thesis:** every recommendation must be traceable back to the original
stakeholder signals, the supporting evidence, the frameworks applied, and a confidence
score. Traceability is a data-model invariant, not a feature.

The first four stages of the core loop (signals → analysis → features → conflicts →
decisions) are implemented and production-wired.

---

## 2. Current Architecture

Layered, dependency-injected, and contract-first. Each AI stage is a thin subclass of a
shared runtime harness; everything off the LLM path is deterministic and unit-tested.

```
HTTP (FastAPI)
  └─ Routes + Pydantic schemas        app/api/
       └─ Services (transactions, use cases)   app/services/
            ├─ Repositories (persistence)       app/repositories/
            │     └─ SQLAlchemy models + Alembic app/models/, alembic/
            └─ AI Stages (runner per stage)      app/stages/
                  └─ Runtime Harness (orchestration, retries, metrics) app/ai_runtime/
                        └─ AI Contracts (Pydantic) + Validators app/ai_contracts/
                              └─ Claude Adapter (ToolCallClient)  app/ai_clients/
Cross-cutting: Observability (structured logging, correlation ids) app/observability/
```

**Key boundaries**
- The **LLM is untrusted**: every model output crosses a schema gate and a stage-specific
  semantic gate before it becomes state.
- The Anthropic SDK is isolated behind a single `ToolCallClient` protocol, so the entire
  system runs in tests with no network and no API key.
- Vectors are designated for ChromaDB (handles reserved via `chroma_id`); the relational
  store is the system of record.

---

## 3. Completed Stages

| Stage | Capability | Trust gate | Status |
|---|---|---|---|
| **Stage 1** | Signal Analysis — parse one signal into intent, stakeholder type, urgency, sentiment, source-anchored claims | **Grounding** (claim spans must substantiate the claim) | ✅ Complete |
| **Stage 2** | Feature Extraction — cluster grounded signals into normalized features with JTBD | **Provenance** (every feature cites real signals; inputs fully partitioned) | ✅ Complete |
| **Stage 3** | Conflict Detection — surface stakeholder disagreements (priority / risk / resource / strategic) over features | **Conflict integrity** (evidence-traceable + genuine opposition) | ✅ Complete |
| **Stage 4** | Decision Synthesis — convert features + conflicts + evidence into ranked, evidence-backed decisions (recommendation + rationale) | **Decision integrity** (real subject/evidence/conflicts; every conflict over the subject acknowledged) | ✅ Complete |

Supporting infrastructure complete: Runtime Harness, Claude Adapter, Structured Logging,
Alembic migrations, three-phase transaction boundaries.

**Phase 5 — Decision Explainability (`/why`)** ✅ Complete. A read-only provenance projection:
`GET /decisions/{id}/why` traverses decision → direct evidence signals → subject feature →
acknowledged conflicts → original stakeholder inputs (raw text + grounded claims). Built from a
pure, deterministic `ExplanationAssembler` (signal dedup, `reached_via` tagging, `quoted_text`
from source spans, stable ordering) plus a deterministic integrity check that surfaces — never
silently drops — any unresolved reference. No new tables, no LLM, no writes.

---

## 4. Database Schema

Nine tables. UUID primary keys (`sa.Uuid`, native on Postgres, `CHAR(32)` on SQLite).
Enums store lowercase values via `values_callable`. JSON columns hold structured AI
output. Vectors live in ChromaDB (`chroma_id` reserved, nullable).

| Table | Purpose | Key columns | Constraints / indexes |
|---|---|---|---|
| `signal` | Immutable stakeholder signal (provenance root) | `source_type`, `raw_text`, `content_hash`, `chroma_id?`, `workspace_id?` | `UNIQUE(content_hash)`; ORM `before_update` guard (immutability) |
| `parsed_signal` | Grounded Stage 1 analysis (1:1 with signal) | `intent`, `stakeholder_type`, `urgency`, `sentiment`, `extracted_claims` (JSON), `confidence` (JSON), `model_meta` (JSON) | `UNIQUE(signal_id)`, FK→signal `CASCADE` |
| `feature` | Normalized product feature | `title`, `description`, `jtbd`, `status`, `confidence` (JSON), `model_meta` (JSON), `workspace_id?` | PK |
| `feature_signal` | Provenance edge feature↔signal | `relationship`, `created_at` | composite PK `(feature_id, signal_id)`; FK→feature `CASCADE`, FK→signal **`RESTRICT`**; `ix_feature_signal_signal_id` |
| `conflict` | Detected disagreement over a subject | `subject_type`, `subject_id` (polymorphic), `conflict_type`, `severity`, `status`, `confidence` (JSON), `model_meta` (JSON) | PK; `ix_conflict_subject_id` |
| `conflict_party` | One stakeholder position in a conflict | `stakeholder_type`, `stance`, `summary`, `evidence_signal_ids` (JSON) | FK→conflict `CASCADE`; `ix_conflict_party_conflict_id` |
| `decision` | Synthesized, evidence-backed decision over a subject | `subject_type`, `subject_id` (polymorphic), `recommendation`, `title`, `rationale`, `priority_rank`, `status`, `confidence` (JSON), `model_meta` (JSON) | PK; `ix_decision_subject_id` |
| `decision_evidence` | Provenance edge decision↔signal | `created_at` | composite PK `(decision_id, signal_id)`; FK→decision `CASCADE`, FK→signal **`RESTRICT`**; `ix_decision_evidence_signal_id` |
| `decision_conflict` | Edge decision↔acknowledged conflict | `created_at` | composite PK `(decision_id, conflict_id)`; FK→decision `CASCADE`, FK→conflict **`RESTRICT`**; `ix_decision_conflict_conflict_id` |

**Reserved (not yet populated):** `workspace_id` (all tables), `chroma_id` (`signal`).

Stage 4 strengthens evidence traceability relative to Stage 3: a decision's evidence and
its acknowledged conflicts are real **edge tables with FK `RESTRICT`** (deletion
protection), not JSON — so neither a backing signal nor an acknowledged conflict can be
deleted out from under a decision.

---

## 5. API Endpoints

| Method | Path | Purpose | Notable responses |
|---|---|---|---|
| `POST` | `/signals` | Create a signal (idempotent by content hash) | `201` new / `200` existing |
| `GET` | `/signals/{id}` | Retrieve a signal | `404` if missing |
| `POST` | `/signals/{id}/analyze` | Run Stage 1; persist grounded analysis | `422` (structured) if no grounded result |
| `GET` | `/signals/{id}/analysis` | Retrieve the persisted analysis | `404` if not analyzed |
| `POST` | `/features/extract` | Run Stage 2 over analyzed signals | `422` if extraction fails / signals unanalyzed |
| `GET` | `/features/{id}` | Retrieve a feature (with source signals) | `404` |
| `GET` | `/features` | List features (`?signal_id`, `?workspace_id`) | — |
| `POST` | `/conflicts/detect` | Run Stage 3 over features | `404` missing feature / `422` detection fails |
| `GET` | `/conflicts/{id}` | Retrieve a conflict (with positions + evidence) | `404` |
| `GET` | `/conflicts` | List conflicts (`?subject_id`, `?workspace_id`) | — |
| `POST` | `/decisions/synthesize` | Run Stage 4 over features (+ their conflicts) | `404` missing feature / `422` synthesis fails |
| `GET` | `/decisions/{id}` | Retrieve a decision (with evidence + acknowledged conflicts) | `404` |
| `GET` | `/decisions/{id}/why` | Explainability (Phase 5) — provenance walk decision → signals | `404` missing decision |
| `GET` | `/decisions` | List decisions (`?subject_id`, `?workspace_id`) | — |

All requests carry a correlation id (`X-Request-ID`, generated if absent). Stage runners
are read from `app.state.stage{1,2,3,4}_runner` (configured at startup; `503` if absent).
Entry point: `uvicorn app.main:app`.

---

## 6. Runtime Architecture

The harness (`app/ai_runtime/`) owns one orchestration loop shared by every stage:

```
build prompt → call model (forced tool use)
  → tool called?  → schema-validate (Pydantic)  → semantic-validate (stage gate)
  → success: collect confidence + metrics → StageResult
  on retryable failure: append deterministic feedback → RetryPolicy.decide() → retry
```

- **`BaseStageRunner[TContract, TContext]`** — abstract; subclasses supply identity,
  prompts, and validators only.
- **`ToolCallClient`** (protocol) — the single LLM boundary; `ClaudeToolClient` implements
  it with forced tool use, timeout handling, and retryable-vs-fatal error classification.
- **`RetryPolicy` / `RetryDecision`** — deterministic (no jitter), capped exponential
  backoff; the harness is the single retry authority (SDK retries set to 0).
- **`StageResult[T]` / `StageMetrics` / `AttemptMetrics`** — typed outcome with per-attempt
  tokens, stop reasons, validation flags, and confidence.
- **`HarnessManagedMetadataRunner`** (mixin) — strips `model_meta`/`schema_version` from the
  tool the model sees and injects accurate values from the real response (used by Stages 3–4;
  Stages 1–2 inline the same behavior — see Tech Debt).

---

## 7. Validation Architecture

Two-tier per stage: **schema** (structural, in the contract) and **semantic** (stage-specific,
deterministic, dependency-free). Each semantic gate has a pure function returning a
structured report plus a thin adapter implementing the `StageValidator` protocol.

| Stage | Deterministic gate (`app/ai_contracts/validation/`) | What it guarantees |
|---|---|---|
| 1 | `grounding.py` | Each claim's char span substantiates it (token-overlap ≥ threshold); ≥1 grounded claim |
| 2 | `provenance.py` | Every feature cites real input signals; inputs fully partitioned (no fabricated/duplicated/dropped signal) |
| 3 | `conflict_integrity.py` | Subject is a real feature; evidence ⊆ input signals; positions match stakeholders; **genuine opposition** required; empty result allowed |
| 4 | `decision_integrity.py` | Subject is a real feature; evidence ⊆ input signals; acknowledged conflicts are real and about the subject; **every conflict over the subject acknowledged**; no duplicate subjects |

A failed gate produces deterministic retry feedback (field + hint), so the same failure
always yields the same correction message — replayable for research.

---

## 8. Observability Architecture

`app/observability/logging.py` + `app/api/middleware.py`.

- **Structured JSON logs** — `JsonFormatter` renders one JSON line per record, including any
  `extra=` fields and bound context.
- **Correlation IDs** — `CorrelationIdMiddleware` (pure ASGI) honors inbound `X-Request-ID`
  or generates one, binds it to a contextvar via `ContextFilter`, and echoes it on the
  response. Per-request `http_request` log with status + duration.
- **Logged events** — `stage_started`, `stage_succeeded`, `stage_failed`, `stage_retry`
  (validation/retry events), `db_write`, `signal_created`, `signal_deduplicated`,
  `validation_failed`, `http_request`.
- **Testable** — formatter, filter, and context are unit-tested; service/middleware logs are
  asserted via `caplog`. Secrets never logged (`api_key` is a `SecretStr`).
- `configure_logging()` is installed at startup by `create_production_app()` (idempotent,
  non-clobbering).

---

## 9. Current Test Count

**276 tests passing** (1 warning: pre-existing Starlette `TestClient`/httpx deprecation,
unrelated to project code). Pure unit + integration; no network, no API key, deterministic.
Run: `python -m pytest`.

---

## 10. Current Alembic Revision

**Head: `0005_decision_and_edges`.**

| Revision | Adds |
|---|---|
| `0001_signal_parsed_signal` | `signal`, `parsed_signal` |
| `0002_feature_feature_signal` | `feature`, `feature_signal` |
| `0003_feature_signal_signal_id_index` | `ix_feature_signal_signal_id` |
| `0004_conflict_conflict_party` | `conflict`, `conflict_party` (+ indexes) |
| `0005_decision_and_edges` | `decision`, `decision_evidence`, `decision_conflict` (+ indexes) |

Migration↔model parity is enforced by a test comparing columns, types, nullability, PK,
unique constraints, FKs (with `ondelete`), and indexes across all nine tables. Apply with
`alembic upgrade head` (Postgres via `DATABASE_URL`; needs a driver, e.g. `psycopg[binary]`).

---

## 11. Open Technical Debt

| Item | Severity | Notes |
|---|---|---|
| Harness-metadata override duplicated in Stage 1 & 2 runners | Low | `HarnessManagedMetadataRunner` mixin exists and is used by Stages 3 & 4; Stages 1–2 not yet retrofitted (frozen + passing) |
| Conflict evidence stored as JSON (no FK) | Low/Med | `conflict_party.evidence_signal_ids` is JSON, not an FK edge table; validated at write time but no DB-level `RESTRICT`. Trade-off to honor the two-table design (Stage 4 evidence/acknowledgment *do* use FK edge tables — see ADR-012) |
| `conflict.subject_id` / `decision.subject_id` polymorphic (no FK) | Low | Indexed; validity enforced by the integrity gate, not the DB |
| No workspace scoping | Med | `workspace_id` reserved/nullable everywhere; no `Workspace` entity, so stages take explicit id lists rather than "everything in workspace X" |
| Extract/detect/synthesize not idempotent | Med | Re-running `extract_features` / `detect_conflicts` / `synthesize_decisions` creates duplicate rows (no upsert); semantics need a decision |
| Severity / recommendation / priority_rank are model-provided | Low | Not deterministically recomputed from a rubric (the envisioned RICE recomputation is a separate roadmap item); schema-bounded |
| Conflict acknowledgment is structural, not semantic | Low | Stage 4 enforces that a conflict id is *referenced* (deterministic), not that the rationale genuinely engages it; a softer review/eval concern (ADR-012) |
| Non-deterministic child ordering | Low | `feature_signals` / `conflict.parties` have no `ORDER BY`; response ordering varies — minor reproducibility concern |
| Signal immutability is ORM-only | Low | Enforced by a `before_update` event, not a DB trigger (migration intentionally does not add one) |
| Vector/RAG layer absent | Med | `chroma_id` reserved; no embeddings, no framework knowledge base yet |
| Tests run on SQLite | Low | Postgres-specific enum/JSONB behavior not exercised in CI; parity test mitigates structural drift |
| `JSON` not `JSONB` on Postgres | Low | Models use `sa.JSON`; switching to `JSONB` (for indexing/containment) is a future migration |

No Critical or High debt is open. The last hardening sprint closed: missing reverse-provenance
index, LLM-call-inside-transaction, and absence of structured logging.

---

## 12. Remaining Roadmap

Ordered to reach the product thesis (a defensible, traceable decision) fastest.

**Done — the keystone of the decision loop**
- ✅ **Decision Synthesis (Stage 4)** — ranked, evidence-backed decisions that acknowledge
  every conflict over their subject; the conflict-acknowledgment invariant (ADR-012) makes
  silent omission impossible. Persisted with FK-protected evidence/acknowledgment edges.
- ✅ **Decision Explainability (Phase 5)** — `GET /decisions/{id}/why`: the provenance walk back to
  raw signals, with a deterministic integrity check and `quoted_text` per claim. The product's
  signature feature; the Stage 4 edge tables made it a straightforward relational projection.

**P0 — finish the decision loop**
1. **Scoring (RICE)** — deterministic compute in code; LLM only estimates inputs with confidence.
   (Stage 4 ranks via a model-provided `priority_rank`; RICE replaces that with a rubric.)
2. **Confidence service** — compute Evidence Coverage / Reasoning Quality / Input Confidence
   (largely deterministic), capped by open conflict severity.

**P1 — credibility & enterprise polish**
5. **Knowledge / RAG layer** — ChromaDB + curated framework corpus (RICE/JTBD/MoSCoW/strategy/PRD), citations on decisions.
6. **Workspace entity + scoping** — first-class grouping so stages operate over a workspace.
7. **Decision history / audit log** — append-only; Product-Ops "show your work".
8. **Eval harness** — No-RAG vs Framework-RAG vs Signal-RAG metrics (faithfulness, grounding accuracy, decision quality) — product hardening + research-paper backbone.

**P2 — nice to have**
9. Human-in-the-loop override + reason capture; one-page decision brief export; hybrid retrieval; prompt/embedding caching.

**Explicitly out of scope:** auth, billing, notifications, real tool connectors (Slack/Jira/Zendesk), competitive intelligence.
