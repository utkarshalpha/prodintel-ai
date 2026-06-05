"""DecisionExplanationService -- the application service for Phase 5 (``/why``).

Orchestrates a read-only traversal of a decision's provenance graph and delegates
the transformation to the pure :class:`ExplanationAssembler`. It depends only on
repositories (no SQL of its own), makes no LLM calls, and performs no writes, so it
holds no transaction boundary -- it is a pure read path.

Traversal:

    decision --(decision_evidence)--> signals
    decision --(subject_id, polymorphic)--> subject feature --(feature_signal)--> signals
    decision --(decision_conflict)--> conflicts --(parties.evidence_signal_ids JSON)--> signals

All referenced signals (across the three paths) are batch-loaded with their Stage 1
analysis in one query, so the traversal is bounded and free of N+1.
"""

from __future__ import annotations

import uuid

from app.ai_contracts.enums import SubjectType
from app.api.explanation_assembler import ExplanationAssembler
from app.api.schemas_decision_explanation import DecisionExplanationResponse
from app.repositories.conflict_repository import ConflictRepository
from app.repositories.decision_repository import DecisionRepository
from app.repositories.feature_repository import FeatureRepository
from app.repositories.signal_repository import SignalRepository
from app.services.errors import DecisionNotFoundError

__all__ = ["DecisionExplanationService"]


class DecisionExplanationService:
    """Application service for decision explainability (read-only)."""

    def __init__(
        self,
        decision_repo: DecisionRepository,
        feature_repo: FeatureRepository,
        conflict_repo: ConflictRepository,
        signal_repo: SignalRepository,
    ) -> None:
        self._decisions = decision_repo
        self._features = feature_repo
        self._conflicts = conflict_repo
        self._signals = signal_repo

    def explain(self, decision_id: uuid.UUID) -> DecisionExplanationResponse:
        """Return the full provenance explanation for a decision.

        Raises :class:`DecisionNotFoundError` (HTTP 404) if the decision is absent.
        """

        decision = self._decisions.get_with_provenance(decision_id)
        if decision is None:
            raise DecisionNotFoundError(decision_id)

        # Subject feature (polymorphic subject_id; only resolved when it is a feature).
        subject_feature = None
        if decision.subject_type == SubjectType.FEATURE:
            subject_feature = self._features.get_with_signals(decision.subject_id)

        # Acknowledged conflicts (with their parties).
        conflict_ids = [edge.conflict_id for edge in decision.acknowledged_conflicts]
        conflicts = list(self._conflicts.get_many_with_parties(conflict_ids))

        # Gather every referenced signal id across the three provenance paths.
        signal_ids: set[uuid.UUID] = {edge.signal_id for edge in decision.evidence}
        if subject_feature is not None:
            signal_ids.update(fs.signal_id for fs in subject_feature.feature_signals)
        for conflict in conflicts:
            for party in conflict.parties:
                signal_ids.update(uuid.UUID(str(raw)) for raw in party.evidence_signal_ids)

        signals = list(self._signals.list_with_analysis(signal_ids))

        return ExplanationAssembler.assemble(decision, subject_feature, conflicts, signals)
