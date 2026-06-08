"""PipelineService -- the end-to-end orchestration coordinator (saga, not a stage).

Composes the five existing service-layer use cases into one run:

    FeedbackEntry[]
      -> SignalService.create_signal + analyze_signal   (Stage 1, per entry)
      -> FeatureService.extract_features                 (Stage 2)
      -> ConflictService.detect_conflicts                (Stage 3)
      -> DecisionService.synthesize_decisions            (Stage 4, optional grounding)
      -> DecisionExplanationService.explain              (/why, read-only, injected)

Design (deliberately a thin saga coordinator)
---------------------------------------------
* **Transaction-less.** It owns no commit, rollback, or session lifecycle. Each delegated
  service owns its own three-phase (Read -> AI -> Write) transaction boundary (ADR-010).
  A multi-stage AI run *cannot* be one transaction (a DB txn must never be held across a
  model call), so a run is an honest sequence of independently-committed steps. A failure
  mid-run leaves earlier stages **committed** -- there is no cross-stage rollback, by
  design -- and that partial progress is reported, never hidden.
* **No stage logic, ingestion, serialization, LLM construction, API-key handling, HTTP, or
  ``PipelineRun`` persistence.** It receives ready-made runners (engine choice is the
  caller's concern) and returns domain objects.
* **Explanation is injected.** ``DecisionExplanationService`` transitively imports the API
  layer; injecting it (rather than constructing it here) keeps ``app.services`` free of an
  ``app.services -> app.api`` dependency and avoids a circular import. If no explanation
  service is supplied, the explanation step is recorded as skipped.

Canonical output
----------------
:meth:`PipelineService.analyze` returns a :class:`PipelineRunResult` -- never a tuple or
ad-hoc dict -- so Streamlit, an API endpoint, a CLI, or a batch job can inspect *partial*
progress (``completed_stages`` / ``failed_stage`` / ``stage_outcomes``) without the
contract changing. The underlying per-stage ``StageResult`` objects are preserved verbatim
inside :class:`StageOutcome`.

Session-scope contract (important)
----------------------------------
``PipelineRunResult`` carries **live ORM entities** (signals, features, conflicts,
decisions). They are valid only for the lifetime of the caller-owned ``session`` -- read or
serialize them (e.g. via the snapshot assembler) **before the session is closed**. The
showcase runtime and the tests use ``expire_on_commit=False``; an API would serialize within
request scope. *Follow-up (not this phase): migrate the result to detached DTOs to remove
this lifetime coupling.*

Error semantics
---------------
Known service failures (``AnalysisFailedError``, ``FeatureExtractionFailedError``,
``ConflictDetectionFailedError``, ``DecisionSynthesisFailedError``, and the fail-fast corpus
errors) are captured as ``FAILED`` :class:`StageOutcome`s and returned in the result.
**Unexpected exceptions propagate** -- bugs are never swallowed.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from sqlalchemy.orm import Session

from app.ingestion.contract import FeedbackEntry  # re-exported for backwards compatibility
from app.models.conflict import Conflict
from app.models.decision import Decision
from app.models.feature import Feature
from app.models.signal import ParsedSignal, Signal
from app.repositories.conflict_repository import ConflictRepository
from app.repositories.decision_repository import DecisionRepository
from app.repositories.feature_repository import FeatureRepository, FeatureSignalRepository
from app.repositories.parsed_signal_repository import ParsedSignalRepository
from app.repositories.signal_repository import SignalRepository
from app.services.conflict_service import ConflictService
from app.services.decision_service import DecisionService
from app.services.errors import (
    AnalysisFailedError,
    ConflictDetectionFailedError,
    CorpusEmbeddingModelMismatchError,
    DecisionNotFoundError,
    DecisionSynthesisFailedError,
    FeatureExtractionFailedError,
)
from app.services.feature_service import FeatureService
from app.services.signal_service import SignalService
from app.stages.stage4.prompts import FrameworkPassage
from app.vector.errors import EmbeddingDimensionMismatchError

if TYPE_CHECKING:  # typing only -- avoids importing the API-coupled explanation module
    from app.api.schemas_decision_explanation import DecisionExplanationResponse

__all__ = [
    "FeedbackEntry",
    "PipelineStage",
    "StageStatus",
    "StageOutcome",
    "PipelineRunResult",
    "PipelineService",
    "SupportsExplain",
]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@runtime_checkable
class SupportsExplain(Protocol):
    """Structural type for the read-only ``/why`` explainer (injected)."""

    def explain(self, decision_id: uuid.UUID): ...  # -> DecisionExplanationResponse


class PipelineStage(str, Enum):
    """The logical stages of one orchestration run."""

    SIGNAL_ANALYSIS = "signal_analysis"
    FEATURE_EXTRACTION = "feature_extraction"
    CONFLICT_DETECTION = "conflict_detection"
    DECISION_SYNTHESIS = "decision_synthesis"
    EXPLANATION = "explanation"


class StageStatus(str, Enum):
    """Outcome of one stage within a run."""

    SUCCEEDED = "succeeded"   # ran clean
    PARTIAL = "partial"       # signal_analysis only: some entries analyzed, some failed
    FAILED = "failed"         # a hard failure (fatal to the run, except per-entry Stage 1)
    SKIPPED = "skipped"       # not attempted (an earlier stage failed, or not configured)


@dataclass(frozen=True)
class StageOutcome:
    """What happened at one stage, with the underlying StageResult(s) preserved.

    ``stage_results`` holds the real ``StageResult`` objects: one for stages 2-4, *N* for
    signal analysis (one per attempted entry), and none for the read-only explanation step.
    """

    stage: PipelineStage
    status: StageStatus
    detail: str | None
    error_code: str | None
    stage_results: tuple[object, ...]  # StageResult objects (kept loose, as the services do)


@dataclass(frozen=True)
class PipelineRunResult:
    """Canonical, immutable orchestration output (never a tuple or ad-hoc dict).

    Carries live ORM entities (valid within the caller's session scope -- see module
    docstring) plus full partial-run visibility.
    """

    run_id: uuid.UUID
    started_at: datetime
    completed_at: datetime
    # --- domain entities ---
    signals: tuple[Signal, ...]
    parsed_signals: tuple[ParsedSignal, ...]
    features: tuple[Feature, ...]
    unassigned_signal_ids: tuple[uuid.UUID, ...]
    conflicts: tuple[Conflict, ...]
    decisions: tuple[Decision, ...]
    framework_pool: tuple[FrameworkPassage, ...]
    explanations: tuple["DecisionExplanationResponse", ...]  # 0..N; v1 explains the rank-1 decision
    # --- progress / diagnostics ---
    stage_outcomes: tuple[StageOutcome, ...]
    completed_stages: tuple[PipelineStage, ...]
    failed_stage: PipelineStage | None

    @property
    def succeeded(self) -> bool:
        """True iff no stage failed fatally (a partial Stage 1 is still a success)."""

        return self.failed_stage is None


def _error_code(stage_result: object | None) -> str | None:
    error = getattr(stage_result, "error", None)
    code = getattr(error, "code", None)
    return getattr(code, "value", None)


def _failed_outcome(stage: PipelineStage, exc: Exception) -> StageOutcome:
    stage_result = getattr(exc, "stage_result", None)
    return StageOutcome(
        stage=stage,
        status=StageStatus.FAILED,
        detail=str(exc),
        error_code=_error_code(stage_result),
        stage_results=(stage_result,) if stage_result is not None else (),
    )


def _skipped(stages: Sequence[PipelineStage]) -> list[StageOutcome]:
    return [
        StageOutcome(stage=s, status=StageStatus.SKIPPED, detail="skipped (an earlier stage failed)",
                     error_code=None, stage_results=())
        for s in stages
    ]


class PipelineService:
    """End-to-end saga coordinator over the five stage services (transaction-less)."""

    def __init__(
        self,
        session: Session,
        *,
        stage1_runner,
        stage2_runner,
        stage3_runner,
        stage4_runner,
        retrieval_service=None,
        explanation_service: "SupportsExplain | None" = None,
    ) -> None:
        self._session = session
        self._signals = SignalService(
            session, SignalRepository(session), ParsedSignalRepository(session), stage1_runner
        )
        self._features = FeatureService(
            session, FeatureRepository(session), FeatureSignalRepository(session),
            ParsedSignalRepository(session), SignalRepository(session), stage2_runner,
        )
        self._conflicts = ConflictService(
            session, ConflictRepository(session), FeatureRepository(session),
            ParsedSignalRepository(session), stage3_runner,
        )
        self._decisions = DecisionService(
            session, DecisionRepository(session), ConflictRepository(session),
            FeatureRepository(session), ParsedSignalRepository(session), stage4_runner,
            retrieval_service=retrieval_service,
        )
        self._explanation = explanation_service

    def analyze(
        self,
        entries: Sequence[FeedbackEntry],
        *,
        workspace_id: uuid.UUID | None = None,
        corpus_version: str | None = None,
    ) -> PipelineRunResult:
        """Run the full pipeline over the entries, returning a :class:`PipelineRunResult`.

        Raises :class:`ValueError` for empty input. Known service failures are captured in
        the result (with ``failed_stage`` set); unexpected exceptions propagate.
        """

        if not entries:
            raise ValueError("analyze() requires at least one feedback entry")

        run_id = uuid.uuid4()
        started_at = _utcnow()
        outcomes: list[StageOutcome] = []
        completed: list[PipelineStage] = []

        signals: list[Signal] = []
        parsed: list[ParsedSignal] = []
        features: list[Feature] = []
        unassigned: list[uuid.UUID] = []
        conflicts: list[Conflict] = []
        decisions: list[Decision] = []
        framework_pool: tuple[FrameworkPassage, ...] = ()
        explanations: list["DecisionExplanationResponse"] = []

        def _finish(failed_stage: PipelineStage | None) -> PipelineRunResult:
            return PipelineRunResult(
                run_id=run_id,
                started_at=started_at,
                completed_at=_utcnow(),
                signals=tuple(signals),
                parsed_signals=tuple(parsed),
                features=tuple(features),
                unassigned_signal_ids=tuple(unassigned),
                conflicts=tuple(conflicts),
                decisions=tuple(decisions),
                framework_pool=tuple(framework_pool),
                explanations=tuple(explanations),
                stage_outcomes=tuple(outcomes),
                completed_stages=tuple(completed),
                failed_stage=failed_stage,
            )

        # ---- Stage 1: per-entry create + analyze (each entry independent). ------
        s1_results: list[object] = []
        s1_failures = 0
        for entry in entries:
            created = self._signals.create_signal(
                source_type=entry.stakeholder_type, raw_text=entry.text,
                source_ref=entry.source_ref, workspace_id=workspace_id,
            )
            try:
                res = self._signals.analyze_signal(created.signal.id)
            except AnalysisFailedError as exc:
                s1_failures += 1
                if exc.stage_result is not None:
                    s1_results.append(exc.stage_result)
                continue
            signals.append(created.signal)
            parsed.append(res.parsed_signal)
            s1_results.append(res.stage_result)

        if not parsed:
            outcomes.append(StageOutcome(
                PipelineStage.SIGNAL_ANALYSIS, StageStatus.FAILED,
                f"0/{len(entries)} signals analyzed", None, tuple(s1_results)))
            outcomes.extend(_skipped([
                PipelineStage.FEATURE_EXTRACTION, PipelineStage.CONFLICT_DETECTION,
                PipelineStage.DECISION_SYNTHESIS, PipelineStage.EXPLANATION]))
            return _finish(PipelineStage.SIGNAL_ANALYSIS)

        outcomes.append(StageOutcome(
            PipelineStage.SIGNAL_ANALYSIS,
            StageStatus.SUCCEEDED if s1_failures == 0 else StageStatus.PARTIAL,
            f"{len(parsed)}/{len(entries)} signals analyzed", None, tuple(s1_results)))
        completed.append(PipelineStage.SIGNAL_ANALYSIS)
        analyzed_ids = [p.signal_id for p in parsed]

        # ---- Stage 2: feature extraction. --------------------------------------
        try:
            feat_res = self._features.extract_features(analyzed_ids, workspace_id=workspace_id)
        except FeatureExtractionFailedError as exc:
            outcomes.append(_failed_outcome(PipelineStage.FEATURE_EXTRACTION, exc))
            outcomes.extend(_skipped([
                PipelineStage.CONFLICT_DETECTION, PipelineStage.DECISION_SYNTHESIS,
                PipelineStage.EXPLANATION]))
            return _finish(PipelineStage.FEATURE_EXTRACTION)
        features = list(feat_res.features)
        unassigned = list(feat_res.unassigned_signal_ids)
        outcomes.append(StageOutcome(
            PipelineStage.FEATURE_EXTRACTION, StageStatus.SUCCEEDED,
            f"{len(features)} feature(s)", None, (feat_res.stage_result,)))
        completed.append(PipelineStage.FEATURE_EXTRACTION)
        feature_ids = [f.id for f in features]

        # ---- Stage 3: conflict detection (empty conflicts is success). ---------
        try:
            conf_res = self._conflicts.detect_conflicts(feature_ids, workspace_id=workspace_id)
        except ConflictDetectionFailedError as exc:
            outcomes.append(_failed_outcome(PipelineStage.CONFLICT_DETECTION, exc))
            outcomes.extend(_skipped([PipelineStage.DECISION_SYNTHESIS, PipelineStage.EXPLANATION]))
            return _finish(PipelineStage.CONFLICT_DETECTION)
        conflicts = list(conf_res.conflicts)
        outcomes.append(StageOutcome(
            PipelineStage.CONFLICT_DETECTION, StageStatus.SUCCEEDED,
            f"{len(conflicts)} conflict(s)", None, (conf_res.stage_result,)))
        completed.append(PipelineStage.CONFLICT_DETECTION)

        # ---- Stage 4: decision synthesis (+ optional framework grounding). -----
        try:
            dec_res = self._decisions.synthesize_decisions(
                feature_ids, workspace_id=workspace_id, corpus_version=corpus_version)
        except DecisionSynthesisFailedError as exc:
            outcomes.append(_failed_outcome(PipelineStage.DECISION_SYNTHESIS, exc))
            outcomes.extend(_skipped([PipelineStage.EXPLANATION]))
            return _finish(PipelineStage.DECISION_SYNTHESIS)
        except (CorpusEmbeddingModelMismatchError, EmbeddingDimensionMismatchError) as exc:
            outcomes.append(StageOutcome(
                PipelineStage.DECISION_SYNTHESIS, StageStatus.FAILED,
                f"framework grounding failed: {exc}", type(exc).__name__, ()))
            outcomes.extend(_skipped([PipelineStage.EXPLANATION]))
            return _finish(PipelineStage.DECISION_SYNTHESIS)
        decisions = list(dec_res.decisions)
        framework_pool = dec_res.framework_pool
        outcomes.append(StageOutcome(
            PipelineStage.DECISION_SYNTHESIS, StageStatus.SUCCEEDED,
            f"{len(decisions)} decision(s)", None, (dec_res.stage_result,)))
        completed.append(PipelineStage.DECISION_SYNTHESIS)

        # ---- Explanation: read-only, optional, non-fatal. ----------------------
        if self._explanation is None:
            outcomes.append(StageOutcome(
                PipelineStage.EXPLANATION, StageStatus.SKIPPED,
                "no explanation service configured", None, ()))
        elif not decisions:
            outcomes.append(StageOutcome(
                PipelineStage.EXPLANATION, StageStatus.SKIPPED, "no decisions to explain", None, ()))
        else:
            top = min(decisions, key=lambda d: d.priority_rank)
            try:
                explanations.append(self._explanation.explain(top.id))
            except DecisionNotFoundError as exc:
                outcomes.append(StageOutcome(
                    PipelineStage.EXPLANATION, StageStatus.FAILED,
                    f"explanation failed: {exc}", None, ()))
            else:
                outcomes.append(StageOutcome(
                    PipelineStage.EXPLANATION, StageStatus.SUCCEEDED,
                    f"explained decision (priority #{top.priority_rank})", None, ()))
                completed.append(PipelineStage.EXPLANATION)

        return _finish(None)
