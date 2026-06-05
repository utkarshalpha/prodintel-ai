"""Unit tests for the Stage 1 GroundingValidator adapter."""

from __future__ import annotations

from uuid import uuid4

from app.ai_contracts.base import ModelMeta
from app.stages.stage1.runner import Stage1Context
from app.stages.stage1.validators import GROUNDING_ISSUE_CODE, GroundingValidator

from tests.conftest import make_claim, make_parsed_signal

RAW = "The mobile checkout fails when the user taps pay."
# RAW[4:25] == "mobile checkout fails"
GROUNDED_SPAN = (4, 25)


def _ctx() -> Stage1Context:
    return Stage1Context(signal_id=uuid4(), raw_text=RAW)


def test_all_grounded_passes(model_meta: ModelMeta) -> None:
    claim = make_claim(text="mobile checkout fails", span=GROUNDED_SPAN)
    parsed = make_parsed_signal(model_meta=model_meta, claims=[claim])
    result = GroundingValidator().validate(parsed, _ctx())
    assert result.ok is True
    assert result.issues == ()


def test_single_ungrounded_claim_fails(model_meta: ModelMeta) -> None:
    bad = make_claim(text="a refund was requested", span=(0, 3))  # RAW[0:3] == "The"
    parsed = make_parsed_signal(model_meta=model_meta, claims=[bad])
    result = GroundingValidator().validate(parsed, _ctx())

    assert result.ok is False
    issue = result.errors[0]
    assert issue.code == GROUNDING_ISSUE_CODE
    assert issue.field == "extracted_claims.0.source_span"
    assert issue.hint is not None
    assert issue.context["status"] == "fail_ungrounded"


def test_mixed_reports_only_failing_index(model_meta: ModelMeta) -> None:
    good = make_claim(text="mobile checkout fails", span=GROUNDED_SPAN)
    bad = make_claim(text="a refund was requested", span=(0, 3))
    parsed = make_parsed_signal(model_meta=model_meta, claims=[good, bad])
    result = GroundingValidator().validate(parsed, _ctx())

    assert result.ok is False
    assert len(result.errors) == 1
    assert result.errors[0].field == "extracted_claims.1.source_span"


def test_threshold_is_configurable(model_meta: ModelMeta) -> None:
    # Partial-overlap claim: fails strict, passes lenient.
    claim = make_claim(text="mobile checkout sometimes fails intermittently", span=GROUNDED_SPAN)
    parsed = make_parsed_signal(model_meta=model_meta, claims=[claim])

    assert GroundingValidator(threshold=0.9).validate(parsed, _ctx()).ok is False
    assert GroundingValidator(threshold=0.3).validate(parsed, _ctx()).ok is True
