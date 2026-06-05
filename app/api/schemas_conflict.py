"""HTTP request/response models for the conflict API (Stage 3).

Reuses ``ConfidenceResponse`` and ``RunMeta`` from the signal API schemas. Responses
are built with explicit ``from_row`` constructors and expose every conflict's full,
evidence-traceable position set.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.ai_contracts.enums import ConflictStatus, ConflictType, Stance, StakeholderType, SubjectType
from app.api.schemas import ConfidenceResponse, RunMeta
from app.models.conflict import Conflict

__all__ = [
    "ConflictDetectRequest",
    "ConflictPositionResponse",
    "ConflictResponse",
    "ConflictDetectResponse",
]


class ConflictDetectRequest(BaseModel):
    """Body for ``POST /conflicts/detect``."""

    model_config = ConfigDict(extra="forbid")

    feature_ids: list[uuid.UUID] = Field(..., min_length=1, description="Features to analyze for conflict.")
    workspace_id: uuid.UUID | None = Field(default=None, description="Optional workspace scope (reserved).")


class ConflictPositionResponse(BaseModel):
    """One stakeholder's position, with the evidence backing it."""

    stakeholder: StakeholderType
    stance: Stance
    summary: str
    evidence_signal_ids: list[uuid.UUID]


class ConflictResponse(BaseModel):
    """Representation of a stored conflict, including all positions and evidence."""

    id: uuid.UUID
    workspace_id: uuid.UUID | None
    subject_type: SubjectType
    subject_id: uuid.UUID
    conflict_type: ConflictType
    severity: int
    status: ConflictStatus
    stakeholders: list[StakeholderType]
    positions: list[ConflictPositionResponse]
    evidence_signal_ids: list[uuid.UUID]
    confidence: ConfidenceResponse
    created_at: datetime

    @classmethod
    def from_row(cls, conflict: Conflict) -> "ConflictResponse":
        positions = [
            ConflictPositionResponse(
                stakeholder=party.stakeholder_type,
                stance=party.stance,
                summary=party.summary,
                evidence_signal_ids=[uuid.UUID(s) for s in party.evidence_signal_ids],
            )
            for party in conflict.parties
        ]
        evidence: list[uuid.UUID] = []
        for position in positions:
            for signal_id in position.evidence_signal_ids:
                if signal_id not in evidence:
                    evidence.append(signal_id)
        return cls(
            id=conflict.id,
            workspace_id=conflict.workspace_id,
            subject_type=conflict.subject_type,
            subject_id=conflict.subject_id,
            conflict_type=conflict.conflict_type,
            severity=conflict.severity,
            status=conflict.status,
            stakeholders=[party.stakeholder_type for party in conflict.parties],
            positions=positions,
            evidence_signal_ids=evidence,
            confidence=ConfidenceResponse(
                score=conflict.confidence["score"],
                basis=conflict.confidence["basis"],
                components=conflict.confidence.get("components", {}),
            ),
            created_at=conflict.created_at,
        )


class ConflictDetectResponse(BaseModel):
    """Body for ``POST /conflicts/detect``: detected conflicts and run metadata."""

    conflicts: list[ConflictResponse]
    run: RunMeta
