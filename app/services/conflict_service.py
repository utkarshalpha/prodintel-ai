"""ConflictService -- the application service for Stage 3 (Conflict Detection).

Coordinates detecting conflicts among the stakeholders behind a set of features,
plus retrieval and listing. Owns the transaction boundary and follows the hardened
three-phase pattern (Read -> AI -> Write) so **no database transaction is held
during the Claude call**.

Use cases (PART 5)
------------------
* :meth:`detect_conflicts` -- run Stage 3 over features and their backing signals;
  persist evidence-traceable conflicts.
* :meth:`get_conflict`     -- fetch a conflict (with parties) or raise.
* :meth:`list_conflicts`   -- list conflicts, optionally by subject feature/workspace.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.models.conflict import Conflict
from app.observability.logging import get_logger, log_context
from app.repositories.conflict_repository import ConflictRepository
from app.repositories.feature_repository import FeatureRepository
from app.repositories.parsed_signal_repository import ParsedSignalRepository
from app.services.errors import (
    ConflictDetectionFailedError,
    ConflictNotFoundError,
    FeatureNotFoundError,
    SignalsNotAnalyzedError,
)
from app.stages.stage3.prompts import FeatureForPrompt, SignalForConflict
from app.stages.stage3.runner import Stage3Context, Stage3ConflictRunner

__all__ = ["ConflictService", "ConflictDetectionResult"]

_logger = get_logger(__name__)
_STAGE = "stage3"


@dataclass(frozen=True)
class ConflictDetectionResult:
    """Outcome of a detection run: created conflicts + the run's StageResult."""

    conflicts: list[Conflict]
    stage_result: object  # StageResult[ConflictDetectionContract]


class ConflictService:
    """Application service for conflict detection and management."""

    def __init__(
        self,
        session: Session,
        conflict_repo: ConflictRepository,
        feature_repo: FeatureRepository,
        parsed_repo: ParsedSignalRepository,
        runner: Stage3ConflictRunner,
    ) -> None:
        self._session = session
        self._conflicts = conflict_repo
        self._features = feature_repo
        self._parsed = parsed_repo
        self._runner = runner

    # ----------------------------------------------------------------- detect
    def detect_conflicts(
        self,
        feature_ids: Sequence[uuid.UUID],
        *,
        workspace_id: uuid.UUID | None = None,
    ) -> ConflictDetectionResult:
        """Detect conflicts among the stakeholders behind the given features.

        Raises :class:`FeatureNotFoundError` if a feature is missing,
        :class:`SignalsNotAnalyzedError` if a backing signal lacks analysis, and
        :class:`ConflictDetectionFailedError` if Stage 3 produced no valid,
        evidence-traceable result. An empty result (no conflicts) is a success.
        """

        # ---- Phase 1: READ -- gather features + their backing analyses. --------
        features = []
        signal_ids: list[uuid.UUID] = []
        seen: set[uuid.UUID] = set()
        for feature_id in feature_ids:
            feature = self._features.get(feature_id)
            if feature is None:
                self._session.rollback()
                raise FeatureNotFoundError(feature_id)
            features.append(feature)
            for link in feature.feature_signals:
                if link.signal_id not in seen:
                    seen.add(link.signal_id)
                    signal_ids.append(link.signal_id)

        parsed_rows = {sid: self._parsed.get_by_signal_id(sid) for sid in signal_ids}
        missing = [sid for sid, row in parsed_rows.items() if row is None]
        if missing:
            self._session.rollback()
            raise SignalsNotAnalyzedError(missing)

        feature_views = [
            FeatureForPrompt(feature_id=feature.id, title=feature.title, jtbd=feature.jtbd)
            for feature in features
        ]
        signal_views = [
            SignalForConflict(
                signal_id=row.signal_id,
                stakeholder_type=row.stakeholder_type.value,
                intent=row.intent,
                claims=[claim["text"] for claim in row.extracted_claims],
            )
            for row in parsed_rows.values()
        ]
        self._session.commit()  # release the connection before the AI call

        # ---- Phase 2: AI EXECUTION -- no DB transaction/connection held. --------
        with log_context(feature_count=len(feature_views)):
            _logger.info(
                "stage_started",
                extra={"event": "stage_started", "stage": _STAGE, "feature_count": len(feature_views)},
            )
            context = Stage3Context(features=feature_views, signals=signal_views, workspace_id=workspace_id)
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
                raise ConflictDetectionFailedError(result)

            # ---- Phase 3: WRITE -- fresh transaction. ---------------------------
            model_meta = result.output.model_meta.model_dump(mode="json")
            created: list[Conflict] = []
            for item in result.output.conflicts:
                conflict = Conflict.from_contract(item, model_meta, workspace_id=workspace_id)
                self._conflicts.add(conflict)
                created.append(conflict)
            self._session.commit()
            _logger.info(
                "db_write",
                extra={"event": "db_write", "entity": "conflict", "count": len(created)},
            )
            _logger.info(
                "stage_succeeded",
                extra={
                    "event": "stage_succeeded",
                    "stage": _STAGE,
                    "conflict_count": len(created),
                    "attempts": result.attempts_used,
                    "input_tokens": result.metrics.total_input_tokens,
                    "output_tokens": result.metrics.total_output_tokens,
                },
            )
        return ConflictDetectionResult(conflicts=created, stage_result=result)

    # --------------------------------------------------------------------- get
    def get_conflict(self, conflict_id: uuid.UUID) -> Conflict:
        """Return a conflict or raise :class:`ConflictNotFoundError`."""

        conflict = self._conflicts.get(conflict_id)
        if conflict is None:
            raise ConflictNotFoundError(conflict_id)
        return conflict

    def list_conflicts(
        self,
        *,
        workspace_id: uuid.UUID | None = None,
        subject_id: uuid.UUID | None = None,
        limit: int = 100,
    ) -> Sequence[Conflict]:
        """List conflicts, optionally filtered by subject feature or workspace."""

        return self._conflicts.list(workspace_id=workspace_id, subject_id=subject_id, limit=limit)

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
