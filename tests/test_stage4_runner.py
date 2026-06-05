"""Integration tests for Stage4DecisionRunner via a scripted fake client."""

from __future__ import annotations

from uuid import uuid4

from app.ai_runtime.errors import ErrorCode
from app.ai_runtime.retry_policy import RetryPolicy
from app.ai_runtime.stage_result import StageStatus
from app.stages.stage4.prompts import ConflictForDecision, FeatureForDecision, SignalForDecision
from app.stages.stage4.runner import Stage4Context, build_stage4_runner

from tests.app_helpers import Stage4FakeClient

_FEATURE_ID = uuid4()
_SIGNAL_A, _SIGNAL_B = uuid4(), uuid4()
_CONFLICT_ID = uuid4()


def _ctx(*, with_conflict: bool = True):
    features = [
        FeatureForDecision(feature_id=_FEATURE_ID, title="Enterprise SSO", jtbd="When..., I want..., so I can...")
    ]
    signals = [
        SignalForDecision(signal_id=_SIGNAL_A, stakeholder_type="sales", intent="want SSO", claims=["SSO critical"]),
        SignalForDecision(
            signal_id=_SIGNAL_B, stakeholder_type="engineering", intent="risky", claims=["SSO is high risk"]
        ),
    ]
    conflicts = (
        [
            ConflictForDecision(
                conflict_id=_CONFLICT_ID,
                subject_id=_FEATURE_ID,
                conflict_type="risk",
                severity=4,
                summary="sales (advocate) vs engineering (risk_flag)",
            )
        ]
        if with_conflict
        else []
    )
    return Stage4Context(features=features, signals=signals, conflicts=conflicts)


def test_synthesizes_decision_acknowledging_conflict() -> None:
    runner = build_stage4_runner(Stage4FakeClient(mode="recommend"))
    result = runner.run(_ctx())

    assert result.succeeded is True
    assert result.status is StageStatus.SUCCESS
    assert len(result.output.decisions) == 1
    decision = result.output.decisions[0]
    assert decision.subject_id == _FEATURE_ID
    assert _CONFLICT_ID in decision.acknowledged_conflict_ids
    assert decision.recommendation.value == "build_now"
    assert result.output.model_meta.model_id == "claude-test"
    assert result.output.model_meta.input_tokens == 180


def test_decision_without_conflicts_is_success() -> None:
    runner = build_stage4_runner(Stage4FakeClient(mode="recommend"))
    result = runner.run(_ctx(with_conflict=False))
    assert result.succeeded is True
    assert result.output.decisions[0].acknowledged_conflict_ids == []


def test_fabricated_evidence_is_rejected_and_exhausts() -> None:
    runner = build_stage4_runner(
        Stage4FakeClient(mode="fabricate_evidence"), retry_policy=RetryPolicy(max_attempts=2)
    )
    result = runner.run(_ctx())
    assert result.succeeded is False
    assert result.status is StageStatus.FAILED_EXHAUSTED
    assert result.error.code is ErrorCode.SEMANTIC_VALIDATION_FAILED


def test_ignoring_a_known_conflict_is_rejected() -> None:
    runner = build_stage4_runner(
        Stage4FakeClient(mode="ignore_conflict"), retry_policy=RetryPolicy(max_attempts=1)
    )
    result = runner.run(_ctx())
    assert result.succeeded is False
    assert result.error.code is ErrorCode.SEMANTIC_VALIDATION_FAILED


def test_duplicate_subject_is_rejected() -> None:
    runner = build_stage4_runner(
        Stage4FakeClient(mode="duplicate_subject"), retry_policy=RetryPolicy(max_attempts=1)
    )
    result = runner.run(_ctx())
    assert result.succeeded is False
    assert result.error.code is ErrorCode.SEMANTIC_VALIDATION_FAILED


def test_tool_spec_strips_harness_managed_fields() -> None:
    spec = build_stage4_runner(Stage4FakeClient()).tool_spec()
    assert "model_meta" not in spec.input_schema["properties"]
    assert "decisions" in spec.input_schema["properties"]
