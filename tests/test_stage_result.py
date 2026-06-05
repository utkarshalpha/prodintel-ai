"""Unit tests for StageResult and StageStatus."""

from __future__ import annotations

import pytest

from app.ai_runtime.errors import ErrorCode, StageError, StageExecutionError
from app.ai_runtime.metrics import StageMetrics
from app.ai_runtime.stage_result import StageResult, StageStatus

from tests.runtime_helpers import DemoContract


def _confident(value: int = 7) -> DemoContract:
    return DemoContract(value=value, confidence={"score": 0.9})


def test_success_result_exposes_output() -> None:
    contract = _confident()
    result: StageResult[DemoContract] = StageResult(
        stage_name="demo",
        status=StageStatus.SUCCESS,
        output=contract,
        metrics=StageMetrics(stage_name="demo"),
        attempts_used=1,
    )
    assert result.succeeded is True
    assert result.output_or_raise() is contract


def test_failed_result_raises_on_output_demand() -> None:
    error = StageError(code=ErrorCode.SCHEMA_VALIDATION_FAILED, message="bad", attempt=2)
    result: StageResult[DemoContract] = StageResult(
        stage_name="demo",
        status=StageStatus.FAILED_VALIDATION,
        output=None,
        error=error,
        metrics=StageMetrics(stage_name="demo"),
        attempts_used=2,
    )
    assert result.succeeded is False
    with pytest.raises(StageExecutionError) as exc_info:
        result.output_or_raise()
    assert exc_info.value.error is error


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (ErrorCode.TOOL_NOT_CALLED, StageStatus.FAILED_TOOL_CALL),
        (ErrorCode.MALFORMED_TOOL_INPUT, StageStatus.FAILED_TOOL_CALL),
        (ErrorCode.TRUNCATED_OUTPUT, StageStatus.FAILED_TOOL_CALL),
        (ErrorCode.SCHEMA_VALIDATION_FAILED, StageStatus.FAILED_VALIDATION),
        (ErrorCode.SEMANTIC_VALIDATION_FAILED, StageStatus.FAILED_VALIDATION),
        (ErrorCode.CLIENT_ERROR, StageStatus.FAILED_CLIENT),
    ],
)
def test_status_for_error_mapping(code: ErrorCode, expected: StageStatus) -> None:
    assert StageStatus.for_error(code) is expected
