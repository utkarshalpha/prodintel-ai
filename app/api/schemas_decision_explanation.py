"""HTTP response models for Decision Explainability (Phase 5 -- ``GET /decisions/{id}/why``).

A read-only projection of the decision provenance graph. The contract is the source
of truth for the endpoint shape (contract-first): all models are ``frozen`` and
``extra="forbid"`` -- stricter than the legacy response schemas, deliberately, per the
approved Phase 5 design. ``ConfidenceResponse`` is reused from the signal API schemas.

Shape (normalized; signals are expanded once and referenced by id via ``reached_via``):

    DecisionExplanationResponse
      schema_version
      decision         DecisionSummary
      subject_feature  FeatureSummary | None
      direct_evidence  [EvidenceLink]            decision -> signal (decision_evidence)
      conflicts        [ConflictExplanation]     decision -> conflict (+ parties)
      signals          [SignalProvenance]        original stakeholder inputs (+ analysis)
      integrity        IntegritySummary          deterministic completeness check
      meta             ExplanationMeta
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.ai_contracts.enums import (
    ConflictStatus,
    ConflictType,
    DecisionRecommendation,
    DecisionStatus,
    FeatureStatus,
    Stance,
    StakeholderType,
    SubjectType,
)
from app.api.schemas import ConfidenceResponse

__all__ = [
    "DecisionExplanationResponse",
    "DecisionSummary",
    "FeatureSummary",
    "EvidenceLink",
    "ConflictExplanation",
    "ConflictPartyExplanation",
    "SignalProvenance",
    "AnalysisSummary",
    "ClaimProvenance",
    "IntegritySummary",
    "ExplanationMeta",
    "EXPLANATION_SCHEMA_VERSION",
]

EXPLANATION_SCHEMA_VERSION = "1.0"


class _Frozen(BaseModel):
    """Strict, immutable base for every explanation response model."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ClaimProvenance(_Frozen):
    """A grounded claim with the exact source substring it was anchored to."""

    text: str
    source_span: tuple[int, int]
    quoted_text: str  # raw_text[source_span.start:source_span.end] -- the proof of grounding
    claim_confidence: float


class AnalysisSummary(_Frozen):
    """The Stage 1 analysis behind a signal."""

    intent: str
    stakeholder_type: StakeholderType
    urgency: int
    sentiment: float
    confidence: ConfidenceResponse
    claims: list[ClaimProvenance]


class SignalProvenance(_Frozen):
    """An original stakeholder input, its analysis, and how it backs the decision."""

    id: uuid.UUID
    source_type: StakeholderType
    source_ref: str | None
    raw_text: str
    created_at: datetime
    analysis: AnalysisSummary | None
    reached_via: list[str]  # e.g. ["direct_evidence", "feature:<id>", "conflict:<id>"]


class EvidenceLink(_Frozen):
    """A direct decision -> signal evidence edge (``decision_evidence``)."""

    signal_id: uuid.UUID
    created_at: datetime


class ConflictPartyExplanation(_Frozen):
    """One stakeholder position within an acknowledged conflict."""

    stakeholder_type: StakeholderType
    stance: Stance
    summary: str
    evidence_signal_ids: list[uuid.UUID]


class ConflictExplanation(_Frozen):
    """An acknowledged conflict bearing on the decision, with its positions."""

    id: uuid.UUID
    conflict_type: ConflictType
    severity: int
    status: ConflictStatus
    confidence: ConfidenceResponse
    parties: list[ConflictPartyExplanation]


class FeatureSummary(_Frozen):
    """The subject feature the decision is about."""

    id: uuid.UUID
    title: str
    jtbd: str
    status: FeatureStatus
    confidence: ConfidenceResponse
    source_signal_ids: list[uuid.UUID]


class DecisionSummary(_Frozen):
    """The decision being explained."""

    id: uuid.UUID
    workspace_id: uuid.UUID | None
    subject_type: SubjectType
    subject_id: uuid.UUID
    recommendation: DecisionRecommendation
    title: str
    rationale: str
    priority_rank: int
    status: DecisionStatus
    confidence: ConfidenceResponse
    created_at: datetime


class IntegritySummary(_Frozen):
    """Deterministic completeness check over the assembled provenance graph.

    ``complete`` is ``True`` iff the subject feature resolved (when applicable) and
    every referenced signal id resolved to a real row. Unresolved references are
    surfaced here -- never silently dropped.
    """

    complete: bool
    unresolved_signal_ids: list[uuid.UUID]
    notes: list[str]


class ExplanationMeta(_Frozen):
    """Lightweight metadata about the explanation."""

    decision_id: uuid.UUID
    signal_count: int
    conflict_count: int


class DecisionExplanationResponse(_Frozen):
    """The full, read-only explanation of why a decision was made."""

    schema_version: str = EXPLANATION_SCHEMA_VERSION
    decision: DecisionSummary
    subject_feature: FeatureSummary | None
    direct_evidence: list[EvidenceLink]
    conflicts: list[ConflictExplanation]
    signals: list[SignalProvenance]
    integrity: IntegritySummary
    meta: ExplanationMeta
