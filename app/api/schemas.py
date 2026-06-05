"""HTTP request/response models for the signal API (PART 3).

These live at the HTTP boundary and are intentionally separate from both the ORM
models and the AI contracts. Responses are built with explicit ``from_row``
constructors rather than attribute aliasing, so the mapping from storage to wire is
unambiguous.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.ai_contracts.enums import StakeholderType
from app.ai_runtime.stage_result import StageResult
from app.models.signal import ParsedSignal, Signal

__all__ = [
    "SignalCreateRequest",
    "SignalResponse",
    "ClaimResponse",
    "ConfidenceResponse",
    "AnalysisResponse",
    "RunMeta",
    "AnalyzeResponse",
]


class SignalCreateRequest(BaseModel):
    """Body for ``POST /signals``."""

    model_config = ConfigDict(extra="forbid")

    source_type: StakeholderType = Field(..., description="Origin channel of the signal.")
    raw_text: str = Field(..., min_length=1, description="The stakeholder signal text.")
    source_ref: str | None = Field(default=None, max_length=255, description="External reference, e.g. ticket id.")
    workspace_id: uuid.UUID | None = Field(default=None, description="Reserved; optional workspace scope.")


class SignalResponse(BaseModel):
    """Representation of a stored signal."""

    id: uuid.UUID
    workspace_id: uuid.UUID | None
    source_type: StakeholderType
    raw_text: str
    source_ref: str | None
    content_hash: str
    created_at: datetime

    @classmethod
    def from_row(cls, signal: Signal) -> "SignalResponse":
        return cls(
            id=signal.id,
            workspace_id=signal.workspace_id,
            source_type=signal.source_type,
            raw_text=signal.raw_text,
            source_ref=signal.source_ref,
            content_hash=signal.content_hash,
            created_at=signal.created_at,
        )


class ClaimResponse(BaseModel):
    """A single grounded claim."""

    text: str
    source_span: tuple[int, int]
    claim_confidence: float


class ConfidenceResponse(BaseModel):
    """Decomposed confidence for the analysis."""

    score: float
    basis: str
    components: dict[str, float] = Field(default_factory=dict)


class AnalysisResponse(BaseModel):
    """Persisted Stage 1 analysis of a signal."""

    signal_id: uuid.UUID
    intent: str
    stakeholder_type: StakeholderType
    urgency: int
    sentiment: float
    claims: list[ClaimResponse]
    confidence: ConfidenceResponse
    model_meta: dict
    created_at: datetime

    @classmethod
    def from_row(cls, parsed: ParsedSignal) -> "AnalysisResponse":
        return cls(
            signal_id=parsed.signal_id,
            intent=parsed.intent,
            stakeholder_type=parsed.stakeholder_type,
            urgency=parsed.urgency,
            sentiment=parsed.sentiment,
            claims=[ClaimResponse(**claim) for claim in parsed.extracted_claims],
            confidence=ConfidenceResponse(
                score=parsed.confidence["score"],
                basis=parsed.confidence["basis"],
                components=parsed.confidence.get("components", {}),
            ),
            model_meta=parsed.model_meta,
            created_at=parsed.created_at,
        )


class RunMeta(BaseModel):
    """Metadata about the analysis run (from the StageResult)."""

    status: str
    attempts_used: int
    total_input_tokens: int
    total_output_tokens: int

    @classmethod
    def from_result(cls, result: StageResult) -> "RunMeta":
        return cls(
            status=result.status.value,
            attempts_used=result.attempts_used,
            total_input_tokens=result.metrics.total_input_tokens,
            total_output_tokens=result.metrics.total_output_tokens,
        )


class AnalyzeResponse(BaseModel):
    """Body for ``POST /signals/{id}/analyze``: the analysis plus run metadata."""

    analysis: AnalysisResponse
    run: RunMeta
