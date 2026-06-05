"""DecisionService -- the application service for Stage 4 (Decision Synthesis).

Coordinates synthesizing evidence-backed decisions over a set of features, drawing
on the stakeholder signals behind them and the conflicts already detected over them,
plus retrieval and listing. Owns the transaction boundary and follows the hardened
three-phase pattern (Read -> AI -> Write) so **no database transaction is held
during the Claude call**.

Use cases (PART 5)
------------------
* :meth:`synthesize_decisions` -- run Stage 4 over features, their signals, and their
  conflicts; persist evidence-traceable decisions.
* :meth:`get_decision`         -- fetch a decision (with evidence + conflicts) or raise.
* :meth:`list_decisions`       -- list decisions, optionally by subject feature/workspace.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.models.conflict import Conflict
from app.models.decision import Decision
from app.observability.logging import get_logger, log_context
from app.repositories.conflict_repository import ConflictRepository
from app.repositories.decision_repository import DecisionRepository
from app.repositories.feature_repository import FeatureRepository
from app.repositories.parsed_signal_repository import ParsedSignalRepository
from app.services.errors import (
    DecisionNotFoundError,
    DecisionSynthesisFailedError,
    FeatureNotFoundError,
    SignalsNotAnalyzedError,
)
from app.stages.stage4.prompts import ConflictForDecision, FeatureForDecision, SignalForDecision
from app.stages.stage4.runner import Stage4Context, Stage4DecisionRunner

__all__ = ["DecisionService", "DecisionSynthesisResult"]

_logger = get_logger(__name__)
_STAGE = "stage4"


@dataclass(frozen=True)
class DecisionSynthesisResult:
    """Outcome of a synthesis run: created decisions + the run's StageResult."""

    decisions: list[Decision]
    stage_result: object  # StageResult[DecisionSynthesisContract]


class DecisionService:
    """Application service for decision synthesis and management."""

    def __init__(
        self,
        session: Session,
        decision_repo: DecisionRepository,
        conflict_repo: ConflictRepository,
        feature_repo: FeatureRepository,
        parsed_repo: ParsedSignalRepository,
        runner: Stage4DecisionRunner,
    ) -> None:
        self._session = session
        self._decisions = decision_repo
        self._conflicts = conflict_repo
        self._features = feature_repo
        self._parsed = parsed_repo
        self._runner = runner

    # -------------------------------------------------------------- synthesize
    def synthesize_decisions(
        self,
        feature_ids: Sequence[uuid.UUID],
        *,
        workspace_id: uuid.UUID | None = None,
    ) -> DecisionSynthesisResult:
        """Synthesize evidence-backed decisions over the given features.

        Raises :class:`FeatureNotFoundError` if a feature is missing,
        :class:`SignalsNotAnalyzedError` if a backing signal lacks analysis, and
        :class:`DecisionSynthesisFailedError` if Stage 4 produced no valid,
        evidence-traceable result (nothing is persisted in that case).
        """

        # ---- Phase 1: READ -- gather features, their analyses, and conflicts. ---
        features = []
        signal_ids: list[uuid.UUID] = []
        seen_signals: set[uuid.UUID] = set()
        for feature_id in feature_ids:
            feature = self._features.get(feature_id)
            if feature is None:
                self._session.rollback()
                raise FeatureNotFoundError(feature_id)
            features.append(feature)
            for link in feature.feature_signals:
                if link.signal_id not in seen_signals:
                    seen_signals.add(link.signal_id)
                    signal_ids.append(link.signal_id)

        parsed_rows = {sid: self._parsed.get_by_signal_id(sid) for sid in signal_ids}
        missing = [sid for sid, row in parsed_rows.items() if row is None]
        if missing:
            self._session.rollback()
            raise SignalsNotAnalyzedError(missing)

        # ALL conflicts already detected over any of the subject features. This must
        # not truncate: the decision-integrity gate requires every conflict over a
        # subject to be acknowledged (ADR-012), so a bounded listing would let a
        # decision silently omit real conflicts. list_by_subjects() is unbounded.
        conflicts: list[Conflict] = list(
            self._conflicts.list_by_subjects([feature.id for feature in features])
        )

        feature_views = [
            FeatureForDecision(feature_id=feature.id, title=feature.title, jtbd=feature.jtbd)
            for feature in features
        ]
        signal_views = [
            SignalForDecision(
                signal_id=row.signal_id,
                stakeholder_type=row.stakeholder_type.value,
                intent=row.intent,
                claims=[claim["text"] for claim in row.extracted_claims],
            )
            for row in parsed_rows.values()
        ]
        conflict_views = [
            ConflictForDecision(
                conflict_id=conflict.id,
                subject_id=conflict.subject_id,
                conflict_type=conflict.conflict_type.value,
                severity=conflict.severity,
                summary=_summarize_conflict(conflict),
            )
            for conflict in conflicts
        ]
        self._session.commit()  # release the connection before the AI call

        # ---- Phase 2: AI EXECUTION -- no DB transaction/connection held. --------
        with log_context(feature_count=len(feature_views), conflict_count=len(conflict_views)):
            _logger.info(
                "stage_started",
                extra={
                    "event": "stage_started",
                    "stage": _STAGE,
                    "feature_count": len(feature_views),
                    "conflict_count": len(conflict_views),
                },
            )
            context = Stage4Context(
                features=feature_views,
                signals=signal_views,
                conflicts=conflict_views,
                workspace_id=workspace_id,
            )
            result = self._runner.run(context)
            self._log_stage_attempts(result)

            if not result.succeeded or result.output is None:
                _logger.warning(
                    "stage_failed",
                    extra={
                        "event": "stage_failed",
                        "stage": _STAGE,
                        "status": result.status.value,
                        "error_code": result.error.code.value if result.error else None,
                        "attempts": result.attempts_used,
                    },
                )
                raise DecisionSynthesisFailedError(result)

            # ---- Phase 3: WRITE -- fresh transaction. ---------------------------
            model_meta = result.output.model_meta.model_dump(mode="json")
            created: list[Decision] = []
            for item in result.output.decisions:
                decision = Decision.from_contract(item, model_meta, workspace_id=workspace_id)
                self._decisions.create_with_links(
                    decision, item.evidence_signal_ids, item.acknowledged_conflict_ids
                )
                created.append(decision)
            self._session.commit()
            _logger.info(
                "db_write",
                extra={"event": "db_write", "entity": "decision", "count": len(created)},
            )
            _logger.info(
                "stage_succeeded",
                extra={
                    "event": "stage_succeeded",
                    "stage": _STAGE,
                    "decision_count": len(created),
                    "attempts": result.attempts_used,
                    "input_tokens": result.metrics.total_input_tokens,
                    "output_tokens": result.metrics.total_output_tokens,
                },
            )
        return DecisionSynthesisResult(decisions=created, stage_result=result)

    # --------------------------------------------------------------------- get
    def get_decision(self, decision_id: uuid.UUID) -> Decision:
        """Return a decision or raise :class:`DecisionNotFoundError`."""

        decision = self._decisions.get(decision_id)
        if decision is None:
            raise DecisionNotFoundError(decision_id)
        return decision

    def list_decisions(
        self,
        *,
        workspace_id: uuid.UUID | None = None,
        subject_id: uuid.UUID | None = None,
        limit: int = 100,
    ) -> Sequence[Decision]:
        """List decisions, optionally filtered by subject feature or workspace."""

        return self._decisions.list(workspace_id=workspace_id, subject_id=subject_id, limit=limit)

    # ----------------------------------------------------------------- logging
    @staticmethod
    def _log_stage_attempts(result: object) -> None:
        """Emit one structured log per failed attempt (validation/retry events)."""

        for attempt in result.metrics.attempts:  # type: ignore[attr-defined]
            if attempt.error_code is not None:
                _logger.warning(
                    "stage_retry",
                    extra={
                        "event": "retry_event",
                        "stage": _STAGE,
                        "attempt": attempt.attempt,
                        "error_code": attempt.error_code.value,
                    },
                )


def _summarize_conflict(conflict: Conflict) -> str:
    """Render a one-line, human-readable summary of a conflict from its parties."""

    parties = "; ".join(
        f"{party.stakeholder_type.value} ({party.stance.value}): {party.summary}"
        for party in conflict.parties
    )
    return f"{conflict.conflict_type.value} conflict (severity {conflict.severity}) -- {parties}"
