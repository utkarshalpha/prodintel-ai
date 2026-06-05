"""HTTP request/response models for the feature API (Stage 2).

Reuses ``ConfidenceResponse`` and ``RunMeta`` from the signal API schemas to avoid
duplication. Responses are built with explicit ``from_row`` constructors.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.ai_contracts.enums import FeatureStatus
from app.api.schemas import ConfidenceResponse, RunMeta
from app.models.feature import Feature

__all__ = [
    "FeatureExtractRequest",
    "FeatureResponse",
    "FeatureExtractResponse",
]


class FeatureExtractRequest(BaseModel):
    """Body for ``POST /features/extract``."""

    model_config = ConfigDict(extra="forbid")

    signal_ids: list[uuid.UUID] = Field(..., min_length=1, description="Analyzed signals to cluster.")
    workspace_id: uuid.UUID | None = Field(default=None, description="Optional workspace scope (reserved).")


class FeatureResponse(BaseModel):
    """Representation of a stored feature, including its source signals."""

    id: uuid.UUID
    workspace_id: uuid.UUID | None
    title: str
    description: str
    jtbd: str
    status: FeatureStatus
    confidence: ConfidenceResponse
    source_signal_ids: list[uuid.UUID]
    created_at: datetime

    @classmethod
    def from_row(cls, feature: Feature) -> "FeatureResponse":
        return cls(
            id=feature.id,
            workspace_id=feature.workspace_id,
            title=feature.title,
            description=feature.description,
            jtbd=feature.jtbd,
            status=feature.status,
            confidence=ConfidenceResponse(
                score=feature.confidence["score"],
                basis=feature.confidence["basis"],
                components=feature.confidence.get("components", {}),
            ),
            source_signal_ids=[link.signal_id for link in feature.feature_signals],
            created_at=feature.created_at,
        )


class FeatureExtractResponse(BaseModel):
    """Body for ``POST /features/extract``: features, unassigned ids, and run meta."""

    features: list[FeatureResponse]
    unassigned_signal_ids: list[uuid.UUID]
    run: RunMeta
