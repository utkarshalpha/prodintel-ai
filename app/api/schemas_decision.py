"""HTTP request/response models for the decision API (Stage 4).

Reuses ``ConfidenceResponse`` and ``RunMeta`` from the signal API schemas. Responses
are built with explicit ``from_row`` constructors and expose every decision's full,
evidence-traceable provenance: the signals that substantiate it and the conflicts it
acknowledges.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.ai_contracts.enums import DecisionRecommendation, DecisionStatus, SubjectType
from app.api.schemas import ConfidenceResponse, RunMeta
from app.models.decision import Decision

__all__ = [
    "DecisionSynthesizeRequest",
    "DecisionResponse",
    "DecisionSynthesizeResponse",
]


class DecisionSynthesizeRequest(BaseModel):
    """Body for ``POST /decisions/synthesize``."""

    model_config = ConfigDict(extra="forbid")

    feature_ids: list[uuid.UUID] = Field(..., min_length=1, description="Features to synthesize decisions over.")
    workspace_id: uuid.UUID | None = Field(default=None, description="Optional workspace scope (reserved).")


class DecisionResponse(BaseModel):
    """Representation of a stored decision, with its evidence and acknowledged conflicts."""

    id: uuid.UUID
    workspace_id: uuid.UUID | None
    subject_type: SubjectType
    subject_id: uuid.UUID
    recommendation: DecisionRecommendation
    title: str
    rationale: str
    priority_rank: int
    status: DecisionStatus
    acknowledged_conflict_ids: list[uuid.UUID]
    evidence_signal_ids: list[uuid.UUID]
    confidence: ConfidenceResponse
    created_at: datetime

    @classmethod
    def from_row(cls, decision: Decision) -> "DecisionResponse":
        return cls(
            id=decision.id,
            workspace_id=decision.workspace_id,
            subject_type=decision.subject_type,
            subject_id=decision.subject_id,
            recommendation=decision.recommendation,
            title=decision.title,
            rationale=decision.rationale,
            priority_rank=decision.priority_rank,
            status=decision.status,
            acknowledged_conflict_ids=[edge.conflict_id for edge in decision.acknowledged_conflicts],
            evidence_signal_ids=[edge.signal_id for edge in decision.evidence],
            confidence=ConfidenceResponse(
                score=decision.confidence["score"],
                basis=decision.confidence["basis"],
                components=decision.confidence.get("components", {}),
            ),
            created_at=decision.created_at,
        )


class DecisionSynthesizeResponse(BaseModel):
    """Body for ``POST /decisions/synthesize``: synthesized decisions and run metadata."""

    decisions: list[DecisionResponse]
    run: RunMeta
