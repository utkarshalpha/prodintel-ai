"""Unit tests for the base primitives and the Stage 1 signal contract.

Covers the trust-boundary guarantees that must hold structurally, independent of
any signal text: strictness (no extra fields), immutability, range enforcement,
span ordering, the >= 1 claim rule, and confidence self-consistency.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.ai_contracts.base import (
    ConfidenceBlock,
    ModelMeta,
    confidence_basis_for_score,
)
from app.ai_contracts.enums import ConfidenceBasis, StakeholderType
from app.ai_contracts.stage1_signal import ExtractedClaim, ParsedSignalContract

from tests.conftest import make_claim, make_parsed_signal


# --------------------------------------------------------------------------- #
# confidence_basis_for_score / ConfidenceBlock
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (1.0, ConfidenceBasis.STRONG),
        (0.8, ConfidenceBasis.STRONG),
        (0.79, ConfidenceBasis.MODERATE),
        (0.6, ConfidenceBasis.MODERATE),
        (0.59, ConfidenceBasis.WEAK),
        (0.4, ConfidenceBasis.WEAK),
        (0.39, ConfidenceBasis.INSUFFICIENT),
        (0.0, ConfidenceBasis.INSUFFICIENT),
    ],
)
def test_basis_band_boundaries(score: float, expected: ConfidenceBasis) -> None:
    assert confidence_basis_for_score(score) is expected


def test_basis_clamps_out_of_range_input() -> None:
    # Defensive clamp: never raises, always returns a band.
    assert confidence_basis_for_score(5.0) is ConfidenceBasis.STRONG
    assert confidence_basis_for_score(-2.0) is ConfidenceBasis.INSUFFICIENT


def test_confidence_basis_is_derived_not_trusted() -> None:
    # Caller supplies a deliberately wrong band; contract overwrites it from score.
    block = ConfidenceBlock(score=0.9, basis=ConfidenceBasis.INSUFFICIENT)
    assert block.basis is ConfidenceBasis.STRONG


def test_confidence_basis_injected_when_absent() -> None:
    block = ConfidenceBlock(score=0.5)
    assert block.basis is ConfidenceBasis.WEAK


def test_confidence_score_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        ConfidenceBlock(score=1.5)


def test_confidence_component_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        ConfidenceBlock(score=0.7, components={"bad": 1.2})


def test_confidence_block_is_frozen() -> None:
    block = ConfidenceBlock(score=0.7)
    with pytest.raises(ValidationError):
        block.score = 0.2  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Strictness / immutability (FrozenModel via ModelMeta)
# --------------------------------------------------------------------------- #
def test_extra_field_is_forbidden(model_meta: ModelMeta) -> None:
    with pytest.raises(ValidationError):
        ModelMeta(
            model_id="x",
            prompt_version="v1",
            input_tokens=1,
            output_tokens=1,
            stop_reason="tool_use",
            hallucinated_field="boom",  # type: ignore[call-arg]
        )


def test_model_meta_negative_tokens_rejected() -> None:
    with pytest.raises(ValidationError):
        ModelMeta(
            model_id="x",
            prompt_version="v1",
            input_tokens=-1,
            output_tokens=1,
            stop_reason="tool_use",
        )


# --------------------------------------------------------------------------- #
# ExtractedClaim
# --------------------------------------------------------------------------- #
def test_valid_claim_round_trips() -> None:
    claim = make_claim(text="login is broken", span=(3, 18), confidence=0.8)
    assert claim.source_span == (3, 18)
    assert claim.claim_confidence == 0.8


def test_claim_empty_text_rejected() -> None:
    with pytest.raises(ValidationError):
        ExtractedClaim(text="   ", source_span=(0, 5), claim_confidence=0.5)


def test_claim_span_end_before_start_rejected() -> None:
    with pytest.raises(ValidationError):
        ExtractedClaim(text="x", source_span=(10, 4), claim_confidence=0.5)


def test_claim_span_negative_start_rejected() -> None:
    with pytest.raises(ValidationError):
        ExtractedClaim(text="x", source_span=(-1, 4), claim_confidence=0.5)


def test_claim_span_equal_indices_rejected() -> None:
    with pytest.raises(ValidationError):
        ExtractedClaim(text="x", source_span=(5, 5), claim_confidence=0.5)


def test_claim_confidence_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        ExtractedClaim(text="x", source_span=(0, 2), claim_confidence=1.4)


# --------------------------------------------------------------------------- #
# ParsedSignalContract
# --------------------------------------------------------------------------- #
def test_valid_parsed_signal_round_trips(model_meta: ModelMeta) -> None:
    sig_id = uuid4()
    parsed = make_parsed_signal(model_meta=model_meta, signal_id=sig_id)
    assert parsed.signal_id == sig_id
    assert parsed.schema_version == "1.0"
    assert parsed.stakeholder_type is StakeholderType.CUSTOMER
    assert len(parsed.extracted_claims) == 1


def test_parsed_signal_requires_at_least_one_claim(model_meta: ModelMeta) -> None:
    with pytest.raises(ValidationError):
        make_parsed_signal(model_meta=model_meta, claims=[])


def test_parsed_signal_urgency_out_of_range_rejected(model_meta: ModelMeta) -> None:
    with pytest.raises(ValidationError):
        make_parsed_signal(model_meta=model_meta, urgency=6)


def test_parsed_signal_sentiment_out_of_range_rejected(model_meta: ModelMeta) -> None:
    with pytest.raises(ValidationError):
        make_parsed_signal(model_meta=model_meta, sentiment=-2.0)


def test_parsed_signal_unknown_stakeholder_rejected(model_meta: ModelMeta) -> None:
    with pytest.raises(ValidationError):
        ParsedSignalContract(
            signal_id=uuid4(),
            intent="x",
            stakeholder_type="marketing",  # removed from the product; not a valid enum
            urgency=3,
            sentiment=0.0,
            extracted_claims=[make_claim()],
            confidence={"score": 0.5},
            model_meta=model_meta,
        )


def test_parsed_signal_is_frozen(model_meta: ModelMeta) -> None:
    parsed = make_parsed_signal(model_meta=model_meta)
    with pytest.raises(ValidationError):
        parsed.urgency = 1  # type: ignore[misc]


def test_parsed_signal_json_schema_is_generatable() -> None:
    # The Claude tool input_schema is derived from this; it must be buildable.
    schema = ParsedSignalContract.model_json_schema()
    assert schema["type"] == "object"
    assert "signal_id" in schema["properties"]
    assert "extracted_claims" in schema["properties"]
