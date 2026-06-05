"""FeatureService -- the application service for Stage 2 (Feature Extraction).

Coordinates extracting features from analyzed signals, manual feature creation,
retrieval, and linking. Owns the transaction boundary; delegates persistence to
repositories and clustering to the injected :class:`Stage2FeatureRunner`.

Use cases (PART 5)
------------------
* :meth:`create_feature`        -- manually create a feature with provenance links.
* :meth:`get_feature`           -- fetch a feature or raise.
* :meth:`extract_features`      -- run Stage 2 over analyzed signals; persist features + links.
* :meth:`link_feature_to_signal`-- add a provenance edge between a feature and a signal.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.ai_contracts.enums import Relationship
from app.models.feature import Feature, FeatureSignal
from app.repositories.feature_repository import FeatureRepository, FeatureSignalRepository
from app.repositories.parsed_signal_repository import ParsedSignalRepository
from app.repositories.signal_repository import SignalRepository
from app.observability.logging import get_logger, log_context
from app.services.errors import (
    FeatureExtractionFailedError,
    FeatureNotFoundError,
    SignalNotFoundError,
    SignalsNotAnalyzedError,
)
from app.stages.stage2.prompts import SignalForPrompt
from app.stages.stage2.runner import Stage2Context, Stage2FeatureRunner

__all__ = ["FeatureService", "FeatureExtractionResult"]

_logger = get_logger(__name__)
_STAGE = "stage2"

# Manual-creation confidence/provenance markers (no model involved).
_MANUAL_CONFIDENCE = {"score": 1.0, "basis": "strong", "components": {"manual": 1.0}}
_MANUAL_MODEL_META = {"source": "manual"}


@dataclass(frozen=True)
class FeatureExtractionResult:
    """Outcome of an extraction run: created features + unassigned ids + run info."""

    features: list[Feature]
    unassigned_signal_ids: list[uuid.UUID]
    stage_result: object  # StageResult[FeatureExtractionContract]


class FeatureService:
    """Application service for feature extraction and management."""

    def __init__(
        self,
        session: Session,
        feature_repo: FeatureRepository,
        feature_signal_repo: FeatureSignalRepository,
        parsed_repo: ParsedSignalRepository,
        signal_repo: SignalRepository,
        runner: Stage2FeatureRunner,
    ) -> None:
        self._session = session
        self._features = feature_repo
        self._links = feature_signal_repo
        self._parsed = parsed_repo
        self._signals = signal_repo
        self._runner = runner

    # ----------------------------------------------------------------- extract
    def extract_features(
        self,
        signal_ids: Sequence[uuid.UUID],
        *,
        workspace_id: uuid.UUID | None = None,
    ) -> FeatureExtractionResult:
        """Cluster the given analyzed signals into features and persist them.

        Raises :class:`SignalsNotAnalyzedError` if any signal lacks a Stage 1
        analysis, and :class:`FeatureExtractionFailedError` if Stage 2 produced no
        valid, fully-attributed result (nothing is persisted in that case).
        """

        # ---- Phase 1: READ -- gather analyses, then release the connection. -----
        parsed_rows = {sid: self._parsed.get_by_signal_id(sid) for sid in signal_ids}
        missing = [sid for sid, row in parsed_rows.items() if row is None]
        if missing:
            self._session.rollback()
            _logger.warning(
                "feature_extraction_missing_analyses",
                extra={"event": "validation_failed", "stage": _STAGE, "missing_count": len(missing)},
            )
            raise SignalsNotAnalyzedError(missing)

        views = [
            SignalForPrompt(
                signal_id=row.signal_id,
                stakeholder_type=row.stakeholder_type.value,
                urgency=row.urgency,
                intent=row.intent,
                claims=[claim["text"] for claim in row.extracted_claims],
            )
            for row in parsed_rows.values()
        ]
        self._session.commit()  # end the read transaction before the AI call

        # ---- Phase 2: AI EXECUTION -- no DB transaction/connection held. --------
        with log_context(signal_count=len(views)):
            _logger.info("stage_started", extra={"event": "stage_started", "stage": _STAGE, "signal_count": len(views)})
            context = Stage2Context(signals=views, workspace_id=workspace_id)
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
                raise FeatureExtractionFailedError(result)

            # ---- Phase 3: WRITE -- fresh transaction. ---------------------------
            model_meta = result.output.model_meta.model_dump(mode="json")
            created: list[Feature] = []
            for item in result.output.features:
                feature = Feature.from_contract(item, model_meta, workspace_id=workspace_id)
                self._features.create_with_signals(feature, item.source_signal_ids)
                created.append(feature)
            self._session.commit()
            _logger.info(
                "db_write",
                extra={"event": "db_write", "entity": "feature", "count": len(created)},
            )
            _logger.info(
                "stage_succeeded",
                extra={
                    "event": "stage_succeeded",
                    "stage": _STAGE,
                    "feature_count": len(created),
                    "attempts": result.attempts_used,
                    "input_tokens": result.metrics.total_input_tokens,
                    "output_tokens": result.metrics.total_output_tokens,
                },
            )
        return FeatureExtractionResult(
            features=created,
            unassigned_signal_ids=list(result.output.unassigned_signal_ids),
            stage_result=result,
        )

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

    # ------------------------------------------------------------------ create
    def create_feature(
        self,
        *,
        title: str,
        description: str,
        jtbd: str,
        source_signal_ids: Sequence[uuid.UUID],
        workspace_id: uuid.UUID | None = None,
    ) -> Feature:
        """Manually create a feature linked to existing signals.

        Validates that every source signal exists (raising
        :class:`SignalNotFoundError` otherwise) before persisting.
        """

        for signal_id in source_signal_ids:
            if self._signals.get(signal_id) is None:
                raise SignalNotFoundError(signal_id)

        feature = Feature(
            workspace_id=workspace_id,
            title=title,
            description=description,
            jtbd=jtbd,
            confidence_score=_MANUAL_CONFIDENCE["score"],
            confidence=dict(_MANUAL_CONFIDENCE),
            model_meta=dict(_MANUAL_MODEL_META),
        )
        self._features.create_with_signals(feature, source_signal_ids)
        self._session.commit()
        return feature

    # --------------------------------------------------------------------- get
    def get_feature(self, feature_id: uuid.UUID) -> Feature:
        """Return a feature or raise :class:`FeatureNotFoundError`."""

        feature = self._features.get(feature_id)
        if feature is None:
            raise FeatureNotFoundError(feature_id)
        return feature

    def list_features(
        self,
        *,
        workspace_id: uuid.UUID | None = None,
        signal_id: uuid.UUID | None = None,
        limit: int = 100,
    ) -> Sequence[Feature]:
        """List features, optionally filtered by workspace or linked signal."""

        return self._features.list(workspace_id=workspace_id, signal_id=signal_id, limit=limit)

    # -------------------------------------------------------------------- link
    def link_feature_to_signal(
        self,
        feature_id: uuid.UUID,
        signal_id: uuid.UUID,
        *,
        relationship: Relationship = Relationship.INFORMS,
    ) -> FeatureSignal:
        """Add a provenance edge from a feature to a signal (idempotent).

        Raises if either the feature or the signal does not exist.
        """

        self.get_feature(feature_id)  # 404 if feature missing
        if self._signals.get(signal_id) is None:
            raise SignalNotFoundError(signal_id)

        existing = self._links.get(feature_id, signal_id)
        if existing is not None:
            return existing

        link = self._links.add(
            FeatureSignal(feature_id=feature_id, signal_id=signal_id, relationship_type=relationship)
        )
        self._session.commit()
        return link
