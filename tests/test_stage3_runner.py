"""Integration tests for Stage3ConflictRunner via a scripted fake client."""

from __future__ import annotations

from uuid import uuid4

from app.ai_runtime.errors import ErrorCode
from app.ai_runtime.retry_policy import RetryPolicy
from app.ai_runtime.stage_result import StageStatus
from app.stages.stage3.prompts import FeatureForPrompt, SignalForConflict
from app.stages.stage3.runner import Stage3Context, build_stage3_runner

from tests.app_helpers import Stage3FakeClient


def _ctx():
    feature_id = uuid4()
    signals = [
        SignalForConflict(signal_id=uuid4(), stakeholder_type="sales", intent="want SSO", claims=["SSO critical"]),
        SignalForConflict(
            signal_id=uuid4(), stakeholder_type="engineering", intent="risky", claims=["SSO is high risk"]
        ),
    ]
    features = [FeatureForPrompt(feature_id=feature_id, title="Enterprise SSO", jtbd="When..., I want..., so I can...")]
    return Stage3Context(features=features, signals=signals)


def test_detects_conflict_with_opposition() -> None:
    runner = build_stage3_runner(Stage3FakeClient(mode="conflict"))
    result = runner.run(_ctx())

    assert result.succeeded is True
    assert result.status is StageStatus.SUCCESS
    assert len(result.output.conflicts) == 1
    conflict = result.output.conflicts[0]
    assert {p.stance.value for p in conflict.positions} == {"advocate", "risk_flag"}
    assert result.output.model_meta.model_id == "claude-test"
    assert result.output.model_meta.input_tokens == 160


def test_no_conflict_is_success_with_empty_list() -> None:
    runner = build_stage3_runner(Stage3FakeClient(mode="none"))
    result = runner.run(_ctx())
    assert result.succeeded is True
    assert result.output.conflicts == []


def test_fabricated_evidence_is_rejected_and_exhausts() -> None:
    runner = build_stage3_runner(Stage3FakeClient(mode="fabricate"), retry_policy=RetryPolicy(max_attempts=2))
    result = runner.run(_ctx())
    assert result.succeeded is False
    assert result.status is StageStatus.FAILED_EXHAUSTED
    assert result.error.code is ErrorCode.SEMANTIC_VALIDATION_FAILED


def test_non_opposing_positions_rejected() -> None:
    runner = build_stage3_runner(Stage3FakeClient(mode="no_opposition"), retry_policy=RetryPolicy(max_attempts=1))
    result = runner.run(_ctx())
    assert result.succeeded is False
    assert result.error.code is ErrorCode.SEMANTIC_VALIDATION_FAILED


def test_tool_spec_strips_harness_managed_fields() -> None:
    spec = build_stage3_runner(Stage3FakeClient()).tool_spec()
    assert "model_meta" not in spec.input_schema["properties"]
    assert "conflicts" in spec.input_schema["properties"]
