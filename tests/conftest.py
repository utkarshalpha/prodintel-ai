"""Shared pytest fixtures and factory helpers for the Stage 1 contract tests.

These factories build *valid* contract objects with sensible defaults so each test
can override exactly the one field it cares about, keeping tests focused and
readable.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from app.ai_contracts.base import ModelMeta
from app.ai_contracts.stage1_signal import ExtractedClaim, ParsedSignalContract


@pytest.fixture
def model_meta() -> ModelMeta:
    """A valid :class:`ModelMeta` for embedding in contract roots."""

    return ModelMeta(
        model_id="claude-opus-4-8",
        prompt_version="stage1-v1",
        input_tokens=1200,
        output_tokens=180,
        stop_reason="tool_use",
    )


def make_claim(
    text: str = "checkout fails on mobile",
    span: tuple[int, int] = (0, 24),
    confidence: float = 0.9,
) -> ExtractedClaim:
    """Build a valid :class:`ExtractedClaim` with overridable fields."""

    return ExtractedClaim(text=text, source_span=span, claim_confidence=confidence)


def make_parsed_signal(
    *,
    model_meta: ModelMeta,
    signal_id: UUID | None = None,
    claims: list[ExtractedClaim] | None = None,
    intent: str = "Fix mobile checkout failures",
    urgency: int = 4,
    sentiment: float = -0.4,
    confidence_score: float = 0.82,
) -> ParsedSignalContract:
    """Build a valid :class:`ParsedSignalContract` with overridable fields."""

    return ParsedSignalContract(
        signal_id=signal_id or uuid4(),
        intent=intent,
        stakeholder_type="customer",
        urgency=urgency,
        sentiment=sentiment,
        extracted_claims=claims if claims is not None else [make_claim()],
        confidence={"score": confidence_score, "components": {"span_grounding_ratio": 1.0}},
        model_meta=model_meta,
    )
