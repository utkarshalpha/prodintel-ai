"""Integration tests for Stage2FeatureRunner via a scripted fake client."""

from __future__ import annotations

from uuid import uuid4

from app.ai_runtime.errors import ErrorCode
from app.ai_runtime.retry_policy import RetryPolicy
from app.ai_runtime.stage_result import StageStatus
from app.stages.stage2.prompts import SignalForPrompt
from app.stages.stage2.runner import Stage2Context, build_stage2_runner

from tests.app_helpers import Stage2FakeClient


def _signals(n: int = 2):
    return [
        SignalForPrompt(
            signal_id=uuid4(),
            stakeholder_type="customer",
            urgency=4,
            intent=f"intent {i}",
            claims=[f"claim {i}"],
        )
        for i in range(n)
    ]


def _ctx(signals):
    return Stage2Context(signals=signals)


def test_extraction_success_clusters_all_signals() -> None:
    signals = _signals(2)
    runner = build_stage2_runner(Stage2FakeClient(mode="cluster_all"))
    result = runner.run(_ctx(signals))

    assert result.succeeded is True
    assert result.status is StageStatus.SUCCESS
    assert len(result.output.features) == 1
    feature = result.output.features[0]
    assert set(feature.source_signal_ids) == {s.signal_id for s in signals}
    # model_meta injected from the fake response, not fabricated by the model.
    assert result.output.model_meta.model_id == "claude-test"
    assert result.output.model_meta.input_tokens == 140


def test_fabricated_provenance_is_rejected_and_exhausts() -> None:
    signals = _signals(2)
    runner = build_stage2_runner(Stage2FakeClient(mode="fabricate"), retry_policy=RetryPolicy(max_attempts=2))
    result = runner.run(_ctx(signals))

    assert result.succeeded is False
    assert result.status is StageStatus.FAILED_EXHAUSTED
    assert result.error is not None
    assert result.error.code is ErrorCode.SEMANTIC_VALIDATION_FAILED


def test_dropped_signal_is_rejected() -> None:
    signals = _signals(2)
    runner = build_stage2_runner(Stage2FakeClient(mode="drop_one"), retry_policy=RetryPolicy(max_attempts=1))
    result = runner.run(_ctx(signals))

    assert result.succeeded is False
    assert result.error.code is ErrorCode.SEMANTIC_VALIDATION_FAILED


def test_tool_spec_strips_harness_managed_fields() -> None:
    spec = build_stage2_runner(Stage2FakeClient()).tool_spec()
    props = spec.input_schema["properties"]
    assert "model_meta" not in props
    assert "features" in props
    assert "unassigned_signal_ids" in props
