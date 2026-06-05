# ProdIntel AI — Architecture Decision Record

_Last updated: 2026-06-05_

Each entry records a major engineering decision as **Problem → Decision → Trade-offs →
Alternatives considered**. These are the load-bearing choices behind the system described
in [PROJECT_STATE.md](PROJECT_STATE.md). They are intended to be defensible in an
enterprise review, a portfolio walkthrough, an AI-PM interview, and a research write-up.

| # | Decision | Status |
|---|---|---|
| ADR-001 | Contract-first architecture | Accepted |
| ADR-002 | Harness-managed metadata | Accepted |
| ADR-003 | Deterministic retries | Accepted |
| ADR-004 | Grounding validation (Stage 1) | Accepted |
| ADR-005 | Provenance validation (Stage 2) | Accepted |
| ADR-006 | Conflict integrity validation (Stage 3) | Accepted |
| ADR-007 | Repository pattern | Accepted |
| ADR-008 | Service layer pattern | Accepted |
| ADR-009 | Structured logging & correlation IDs | Accepted |
| ADR-010 | Three-phase transaction boundaries | Accepted |
| ADR-011 | Evidence traceability as a data-model invariant | Accepted |
| ADR-012 | Decision integrity and conflict-acknowledgment invariant | Accepted |

---

## ADR-001 — Contract-first architecture

**Problem.** LLM output is free-form and untrusted. Letting models return prose (or
loosely-shaped JSON) that downstream code parses by hand leads to silent drift,
hallucinated fields polluting state, and untestable boundaries.

**Decision.** Every AI stage is defined by a **Pydantic contract first**. The model is
forced (via tool use) to return exactly that contract; the same contract generates the
tool's `input_schema` (`model_json_schema()`), so the schema the model sees and the
validator the system runs come from one source and cannot diverge. Contracts are strict
(`extra="forbid"`) and immutable (`frozen=True`): an invented field is a hard validation
error, and a validated contract cannot be mutated by a later stage.

**Trade-offs.**
- (+) Single source of truth; type-safe end to end; trivially unit-testable.
- (+) An unexpected field fails loudly instead of corrupting state.
- (−) More upfront ceremony than ad-hoc JSON; contract changes are explicit (a `schema_version`
  literal makes them visible, not silent).

**Alternatives considered.**
- _Free-text + regex/JSON parsing._ Rejected: brittle, untestable, no schema enforcement.
- _JSON mode (no tool)._ Rejected: weaker enforcement than a forced, named tool and no clean
  place for the model to "think" without polluting the structured payload.

---

## ADR-002 — Harness-managed metadata

**Problem.** Contracts carry `model_meta` (model id, token counts, stop reason) for
reproducibility. But those are facts about the _API call_, which the model cannot know —
asking it to emit them invites fabricated provenance.

**Decision.** Treat `model_meta` and `schema_version` as **harness-managed**: strip them from
the tool schema the model sees, then inject accurate values from the real
`LLMToolResponse` before schema validation. Reproducibility metadata is therefore observed,
never self-reported.

**Trade-offs.**
- (+) Provenance is trustworthy; the model can't lie about its own token usage.
- (+) Cleaner prompt (the model isn't asked for fields it can't produce).
- (−) Requires overriding harness extension points (`tool_spec`, `_check_tool_call`,
  `_schema_validate`); a runner becomes single-flight (`_pending_response` is instance state).
  Mitigated by a shared `HarnessManagedMetadataRunner` mixin (one runner per request).

**Alternatives considered.**
- _Model emits `model_meta`._ Rejected: fabricated/incorrect token counts; defeats the purpose.
- _Add `model_meta` after validation._ Rejected: the contract requires it, so validation would
  need a second pass or a relaxed contract — more complex than pre-injection.

---

## ADR-003 — Deterministic retries

**Problem.** Models fail transiently (rate limits, malformed output, failed validation).
Retries are necessary, but non-deterministic retry behavior (jitter, wall-clock dependence,
SDK-internal retries layered under app retries) makes runs irreproducible and hard to reason
about — fatal for a system that also serves as a research artifact.

**Decision.** A single, **deterministic `RetryPolicy`**: retry-or-stop is a pure function of
`(attempt, error_code)`; backoff is capped exponential with **no jitter**. The harness is the
_only_ retry authority — the Anthropic SDK's internal retries are set to `0`. Each retry sends
**deterministic feedback** (the exact violated constraint, with a hint), so the same failure
always yields the same correction.

**Trade-offs.**
- (+) Fully replayable; testable without sleeping (zero-delay default + injected sleeper).
- (+) Clear separation of retryable (`LLMClientError`) vs fatal (`ClaudeFatalError`) failures.
- (−) No randomized backoff means correlated retry storms are possible under load; acceptable
  at current scale, revisit with real traffic.

**Alternatives considered.**
- _SDK built-in retries._ Rejected: opaque, non-deterministic, double-retries under app logic.
- _Jittered backoff._ Rejected: breaks reproducibility; not needed at current concurrency.

---

## ADR-004 — Grounding validation (Stage 1)

**Problem.** A signal analysis that invents claims not present in the source text destroys the
"defensible decision" thesis at the root.

**Decision.** Every extracted claim must carry a `source_span` (char offsets) into the immutable
`raw_text`. A deterministic **grounding gate** verifies bounds, non-emptiness, and token overlap
(≥ threshold) between the claim and the referenced span. Ungrounded claims fail the stage and
trigger a retry; a parse with zero grounded claims is rejected. The persisted analysis therefore
contains nothing the source text does not support.

**Trade-offs.**
- (+) The system _cannot_ persist a hallucinated claim; the strongest single guarantee in the app.
- (+) Pure, dependency-free, adversarially testable; the grounding ratio feeds confidence.
- (−) Token-overlap (Jaccard) is lexical, not semantic — heavy paraphrase can be under-credited;
  mitigated by instructing the model to keep spans exact, and by a tunable threshold.

**Alternatives considered.**
- _Trust the model's claims._ Rejected: no traceability, hallucination risk.
- _Embedding-similarity grounding._ Deferred: heavier, non-deterministic, harder to test; lexical
  overlap is sufficient given exact-span prompting and is replayable.
- _Drop ungrounded claims silently._ Rejected: contracts are immutable, and rejecting + retrying
  yields a stronger guarantee (persisted output is fully grounded) than partial dropping.

---

## ADR-005 — Provenance validation (Stage 2)

**Problem.** Feature extraction clusters many signals into fewer features. The model could cite a
signal that wasn't an input (fabricated provenance), double-count a signal, or silently drop one.

**Decision.** A deterministic **provenance gate**: every feature's `source_signal_ids` must be a
subset of the input set, and the inputs must be **fully partitioned** — each input signal appears
in exactly one feature or in an explicit `unassigned_signal_ids` list, never both, never neither.
Any violation fails the stage and retries. Provenance is also persisted relationally
(`feature_signal` edge table with FK `RESTRICT` toward the signal).

**Trade-offs.**
- (+) Every feature is traceable to real signals; the model must _account for_ every input.
- (+) Forces an explicit "unassigned" decision rather than quiet omission.
- (−) Strict partitioning can reject otherwise-reasonable output that mislabels one signal;
  acceptable because retry feedback is precise and the guarantee is worth it.

**Alternatives considered.**
- _Store `source_signal_ids` as a JSON array on `feature`._ Rejected for Stage 2: loses FK integrity
  and reverse-lookup performance. (Note: Stage 3 _does_ use JSON evidence — see ADR-006/ADR-011.)
- _No partition requirement._ Rejected: allows silent signal loss, undermining traceability.

---

## ADR-006 — Conflict integrity validation (Stage 3)

**Problem.** Conflict detection can fail in more ways than the earlier stages: invent a subject,
cite non-existent evidence, list stakeholders that don't match the positions, or — most subtly —
report _agreement_ as a "conflict."

**Decision.** A deterministic **conflict-integrity gate** enforces: subject is a real input feature;
all evidence ⊆ input signals (every position references evidence); declared stakeholders equal the
positions' stakeholders; and **genuine opposition** exists (≥2 distinct non-neutral stances). An
**empty conflict list is valid** (stakeholders agree) and passes. Violations fail and retry.

**Trade-offs.**
- (+) "No unsupported conflict claims": a conflict without real opposition or real evidence cannot
  be persisted. Empty-is-valid prevents the model from manufacturing drama to seem useful.
- (+) Reuses the established gate-+-adapter pattern; fully deterministic.
- (−) Opposition is defined structurally (distinct non-neutral stances), not semantically — two
  stances could be nominally different yet not truly opposed; acceptable given prompt guidance.

**Alternatives considered.**
- _Deterministically recompute severity from a rubric._ Deferred: severity is model-provided and
  schema-bounded (1–5); recomputation adds complexity not required by the stated rules.
- _Require ≥1 conflict._ Rejected: would force false positives when stakeholders agree.

---

## ADR-007 — Repository pattern

**Problem.** Mixing SQL/session handling into business logic produces untestable services tightly
coupled to the ORM and the database dialect.

**Decision.** Thin, typed **repositories** wrap a SQLAlchemy `Session` and expose intent-named
methods (`get`, `add`, `get_by_content_hash`, `create_with_signals`, `list`, …). They contain **no
business logic** and **flush, never commit** — the service owns the transaction. List queries
eager-load children (`selectinload`) to avoid N+1.

**Trade-offs.**
- (+) Services are testable against fakes or an in-memory DB; persistence concerns are isolated.
- (+) Query patterns (indexes, eager loading) live in one place.
- (−) Some boilerplate; a thin abstraction over the ORM. Justified by testability and clarity.

**Alternatives considered.**
- _Active Record / queries inside services._ Rejected: couples logic to ORM, hurts testability.
- _Generic repository base class._ Deferred: explicit per-aggregate repositories are clearer at this
  size.

---

## ADR-008 — Service layer pattern

**Problem.** Use cases span multiple repositories, the AI runtime, transaction management, and
logging. Without a coordinating layer, this logic leaks into routes or repositories.

**Decision.** A **service per domain area** (`SignalService`, `FeatureService`, `ConflictService`)
owns use cases, the **transaction boundary**, and the dedupe/analysis/detection policy. Services
depend on injected repositories and an injected stage runner — never on the Anthropic SDK directly.
Routes are thin; they translate HTTP ↔ service calls and map service errors to status codes.

**Trade-offs.**
- (+) Clear ownership of transactions and policy; routes stay declarative; DI enables test doubles.
- (+) The Claude client is swappable/absent in tests (runner injected via `app.state`).
- (−) Another layer to navigate; mitigated by a consistent shape across all three services.

**Alternatives considered.**
- _Logic in FastAPI routes._ Rejected: untestable without HTTP, no transaction ownership.
- _Fat repositories._ Rejected: conflates persistence with orchestration.

---

## ADR-009 — Structured logging & correlation IDs

**Problem.** A multi-stage AI system with retries is opaque without observability. Plain text logs
are unsearchable, and there is no way to correlate a request with the stage executions and DB
writes it caused.

**Decision.** **JSON structured logging** (`JsonFormatter`) with a contextvar-based context
(`ContextFilter`) and a pure-ASGI **`CorrelationIdMiddleware`** that binds a per-request correlation
id (honoring inbound `X-Request-ID`). Services emit typed events — `stage_started`,
`stage_succeeded/failed`, `stage_retry` (validation/retry events derived from `StageMetrics` without
touching the frozen harness), `db_write`, `http_request`. Logs are testable (`caplog` + isolated
formatter/filter tests). Secrets are never logged (`api_key` is a `SecretStr`).

**Trade-offs.**
- (+) Every event is searchable and correlatable; retry/validation behavior is visible.
- (+) No harness changes required — the service logs the returned `StageResult`.
- (−) Pure-ASGI middleware (not `BaseHTTPMiddleware`) is slightly more verbose, chosen for reliable
  contextvar propagation to the endpoint.

**Alternatives considered.**
- _`logging` with text formatter._ Rejected: not machine-parseable.
- _`BaseHTTPMiddleware` for correlation._ Rejected: known contextvar-propagation pitfalls across the
  task boundary; pure ASGI is reliable.
- _A third-party structured-logging dependency._ Deferred: stdlib `logging` + a small formatter keeps
  the dependency surface minimal and fully testable.

---

## ADR-010 — Three-phase transaction boundaries

**Problem.** The first implementations read inputs, called Claude (multiple seconds, with backoff),
and wrote results **inside one open database transaction** — pinning a connection (and holding a
transaction) for the entire model latency, which exhausts the pool under concurrency.

**Decision.** Refactor every LLM-bearing service method into **Read → AI → Write**: fetch inputs and
capture plain values, **commit (release the connection)**, run the model with _no_ DB transaction
held, then open a fresh transaction only for the writes. Failure after the read commit persists
nothing.

**Trade-offs.**
- (+) No DB connection is held during model calls; pool pressure scales with DB work, not LLM latency.
- (+) Write phase is a clean, atomic transaction.
- (−) Read and write are no longer one atomic transaction — acceptable here (reads don't need to be
  atomic with writes; a concurrently-deleted input surfaces as an FK error, not corruption).

**Alternatives considered.**
- _One transaction spanning the LLM call._ Rejected: the original problem; unsafe under load.
- _Background job / queue for the AI phase._ Deferred: heavier infrastructure than needed now; the
  three-phase split solves the connection-holding problem without it.

---

## ADR-011 — Evidence traceability as a data-model invariant

**Problem.** "Defensible, evidence-traceable decisions" is the product thesis. If traceability is a
feature bolted on at the end, it will be incomplete and unenforceable.

**Decision.** Make traceability a **data-model and validation invariant**, not a feature. The
immutable `signal` is the provenance root; `parsed_signal` claims are span-anchored to it (ADR-004);
features link to signals via the `feature_signal` edge table with FK `RESTRICT` so a signal backing a
feature **cannot be deleted** (ADR-005); conflicts carry per-position evidence validated against real
signals (ADR-006). Every validated stage output can be walked back to the raw signals that produced it.

**Trade-offs.**
- (+) Traceability is guaranteed by construction; deletion protection is enforced at the DB level for
  features.
- (−) **Conflict evidence is stored as JSON** (`conflict_party.evidence_signal_ids`), not an FK edge
  table — a deliberate trade-off to honor the two-table design for Stage 3. Evidence is validated at
  write time (so conflicts remain traceable), but those references lack DB-level `RESTRICT`. Recorded
  as open tech debt.
- (−) `conflict.subject_id` is polymorphic (feature | objective) and therefore not a single FK; it is
  indexed and validated by the integrity gate.

**Alternatives considered.**
- _Reconstruct provenance from logs/audit after the fact._ Rejected: lossy, unverifiable, not queryable.
- _A third `conflict_party_signal` edge table._ Considered for full FK parity with `feature_signal`;
  deferred to honor the explicit two-table scope for Stage 3, with the trade-off documented.
- _A graph database for the provenance graph._ Rejected at this scale: a well-indexed relational edge
  model answers every traceability query and is far simpler to operate and defend.

---

## ADR-012 — Decision integrity and conflict-acknowledgment invariant

**Problem.** Stage 4 (Decision Synthesis) is the keystone: it converts features, the signals behind
them, and the conflicts already detected over them into a recommendation a PM can defend. The failure
modes are more dangerous than in earlier stages, because the output is the *product*: the model could
decide about a feature that was never submitted, cite a signal that does not exist, "acknowledge" a
conflict that is fictional or that belongs to a different feature, decide the same feature twice with
contradictory recommendations, or — the most insidious failure — **quietly recommend "build now" while
ignoring a severe conflict that was detected over exactly that feature.** A decision that silently
omits a known conflict looks confident and clean, yet it is precisely the kind of unaccountable
recommendation the whole system exists to prevent. "We detected the conflict but the decision never
mentioned it" is disqualifying for a decision-support tool.

**Decision.** Make decision soundness a **deterministic gate** (`validate_decision_integrity`, adapted
by `DecisionIntegrityValidator`), in the same gate-plus-adapter shape as grounding (ADR-004), provenance
(ADR-005), and conflict integrity (ADR-006). Relative to the input feature, signal, and conflict sets, a
synthesis output is sound only when **every** decision satisfies all of:

1. `subject_id` is a real input feature (no fabricated subject);
2. no two decisions share a subject (one decision per feature — no contradictory twins);
3. every evidence id is a real input signal (no unsupported evidence);
4. every acknowledged conflict id is a real input conflict **whose subject is this decision's subject**
   (no fabricated or mis-attributed conflicts); and
5. **every input conflict over the decision's subject is acknowledged** — completeness, not just
   correctness, of conflict accounting.

Rule 5 is the load-bearing invariant. The set of conflicts a decision *must* acknowledge is computed
deterministically from the persisted conflicts whose `subject_id` equals the decision's subject; any
member of that set missing from `acknowledged_conflict_ids` fails the stage and triggers a retry with a
precise, replayable hint. **Silent omission is therefore impossible by construction:** the system
cannot persist a decision that ignores a conflict it already knows about. Acknowledgment is enforced as
a relational fact, too — `decision_conflict` is a real edge table (FK → `conflict` `RESTRICT`), so the
acknowledged conflicts are queryable provenance, not prose buried in a rationale, and an acknowledged
conflict cannot be deleted out from under the decision (extending ADR-011 to Stage 4, alongside the
`decision_evidence` edge with FK → `signal` `RESTRICT`).

**Trade-offs.**
- (+) The strongest guarantee in the product: a persisted decision provably accounts for *all* known
  conflicts over its subject and cites only real evidence. "Show me why" is answerable from the schema.
- (+) Completeness is computed, not trusted — the model cannot earn a pass by acknowledging *some*
  conflicts; it must account for every one or be rejected with a deterministic, replayable correction.
- (+) Reuses the established gate-and-adapter pattern and the harness retry loop verbatim; no new
  control flow, fully unit-testable with no network.
- (−) "Acknowledged" is enforced *structurally* (the conflict id is referenced) rather than
  *semantically* (the rationale genuinely engages with it). A decision could list a conflict id and
  address it only superficially in prose. Acceptable: structural acknowledgment is deterministic and
  forces the conflict into the decision's provenance; prose quality is a softer concern for a later
  review/eval pass, not a gate.
- (−) The invariant assumes conflicts are detected (Stage 3) before synthesis (Stage 4); a feature
  whose conflicts were never detected has an empty "must-acknowledge" set and so is trivially compliant.
  This is correct given the pipeline ordering and is documented, not hidden.

**Alternatives considered.**
- _Let the model decide which conflicts are "relevant."_ Rejected: this is exactly the discretion that
  enables silent omission — the model would rationalize away inconvenient conflicts. Relevance over a
  feature's own conflicts is not a judgment call; all of them must be on the record.
- _Treat unacknowledged conflicts as a warning, not a hard error._ Rejected: a warning does not block
  persistence, so an ignored conflict would still reach the database and the API — defeating the
  invariant. Conflict accounting is a correctness property, not advisory.
- _Store acknowledgments as a JSON array on `decision` (the Stage 3 evidence trade-off)._ Rejected for
  Stage 4: decisions are the product's signature artifact and the basis of the future `/why` walk, so
  acknowledgments earn a real FK edge table with deletion protection rather than unprotected JSON.
- _Require a decision per input feature (full coverage of features, not just conflicts)._ Deferred:
  forcing a decision for every feature is a heavier product rule (some features may legitimately defer);
  the invariant that bites is *conflict* coverage, so that is what the gate enforces now.
