"""Integration tests for BaseStageRunner -- the shared orchestration loop.

Uses a scripted fake client and a trivial demo stage so the full pipeline (call ->
tool-check -> schema -> semantic -> retry -> result) is exercised deterministically
with no network.
"""

from __future__ import annotations

from app.ai_runtime.errors import ErrorCode
from app.ai_runtime.retry_policy import RetryPolicy
from app.ai_runtime.stage_result import StageStatus

from tests.runtime_helpers import (
    DemoContext,
    DemoRunner,
    ScriptedClient,
    client_error,
    make_response,
    valid_input,
)


def _ctx(min_value: int = 0) -> DemoContext:
    return DemoContext(stage_name="demo", min_value=min_value)


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #
def test_success_on_first_attempt() -> None:
    client = ScriptedClient([make_response(valid_input(value=10))])
    result = DemoRunner(client).run(_ctx())

    assert result.succeeded is True
    assert result.status is StageStatus.SUCCESS
    assert result.output is not None and result.output.value == 10
    assert result.attempts_used == 1
    assert result.metrics.total_attempts == 1
    assert result.metrics.total_input_tokens == 100
    assert result.metrics.total_output_tokens == 20
    assert client.call_count == 1


def test_confidence_is_collected_on_success() -> None:
    client = ScriptedClient([make_response(valid_input(value=10, score=0.82))])
    result = DemoRunner(client).run(_ctx())
    assert result.confidence is not None
    assert result.confidence.score == 0.82


# --------------------------------------------------------------------------- #
# Retry paths
# --------------------------------------------------------------------------- #
def test_schema_failure_then_success_appends_feedback() -> None:
    client = ScriptedClient(
        [
            make_response({"value": "not-an-int", "confidence": {"score": 0.8}}),
            make_response(valid_input(value=10)),
        ]
    )
    result = DemoRunner(client).run(_ctx())

    assert result.succeeded is True
    assert result.attempts_used == 2
    # The second call must include the deterministic correction message.
    second_call_messages = client.received[1]
    assert any("rejected" in m.content for m in second_call_messages)


def test_semantic_failure_then_success() -> None:
    client = ScriptedClient(
        [
            make_response(valid_input(value=1)),   # below min_value -> semantic fail
            make_response(valid_input(value=9)),   # passes
        ]
    )
    result = DemoRunner(client).run(_ctx(min_value=5))

    assert result.succeeded is True
    assert result.attempts_used == 2
    # First attempt recorded schema-valid but semantic-invalid.
    assert result.metrics.attempts[0].schema_valid is True
    assert result.metrics.attempts[0].semantic_valid is False
    assert result.metrics.attempts[0].error_code is ErrorCode.SEMANTIC_VALIDATION_FAILED
    # The correction names the minimum.
    assert any("minimum" in m.content for m in client.received[1])


def test_client_error_then_success() -> None:
    client = ScriptedClient([client_error("transient"), make_response(valid_input())])
    result = DemoRunner(client).run(_ctx())

    assert result.succeeded is True
    assert result.attempts_used == 2
    assert result.metrics.attempts[0].error_code is ErrorCode.CLIENT_ERROR
    assert result.metrics.attempts[0].stop_reason == "client_error"


# --------------------------------------------------------------------------- #
# Terminal failures
# --------------------------------------------------------------------------- #
def test_exhaustion_after_repeated_schema_failures() -> None:
    bad = make_response({"value": "x", "confidence": {"score": 0.8}})
    client = ScriptedClient([bad, bad])
    result = DemoRunner(client, RetryPolicy(max_attempts=2)).run(_ctx())

    assert result.succeeded is False
    assert result.status is StageStatus.FAILED_EXHAUSTED
    assert result.error is not None
    assert result.error.code is ErrorCode.SCHEMA_VALIDATION_FAILED
    assert result.attempts_used == 2
    assert result.metrics.succeeded is False


def test_tool_not_called_is_terminal_when_not_retried() -> None:
    client = ScriptedClient([make_response(None, tool_name=None, stop_reason="end_turn")])
    result = DemoRunner(client, RetryPolicy(max_attempts=1)).run(_ctx())

    assert result.status is StageStatus.FAILED_TOOL_CALL
    assert result.error is not None and result.error.code is ErrorCode.TOOL_NOT_CALLED


def test_truncated_output_is_classified() -> None:
    client = ScriptedClient([make_response(None, tool_name=None, stop_reason="max_tokens")])
    result = DemoRunner(client, RetryPolicy(max_attempts=1)).run(_ctx())

    assert result.error is not None and result.error.code is ErrorCode.TRUNCATED_OUTPUT
    assert result.status is StageStatus.FAILED_TOOL_CALL


def test_wrong_tool_name_is_tool_not_called() -> None:
    client = ScriptedClient(
        [make_response(valid_input(), tool_name="some_other_tool", stop_reason="tool_use")]
    )
    result = DemoRunner(client, RetryPolicy(max_attempts=1)).run(_ctx())
    assert result.error is not None and result.error.code is ErrorCode.TOOL_NOT_CALLED


# --------------------------------------------------------------------------- #
# Retry mechanics: backoff actually consults the injected sleeper
# --------------------------------------------------------------------------- #
def test_retry_waits_using_injected_sleeper() -> None:
    slept: list[float] = []
    bad = make_response({"value": "x", "confidence": {"score": 0.8}})
    client = ScriptedClient([bad, make_response(valid_input())])
    policy = RetryPolicy(max_attempts=3, base_delay_seconds=0.25, backoff_multiplier=2.0)

    result = DemoRunner(client, policy, sleep=slept.append).run(_ctx())

    assert result.succeeded is True
    assert slept == [0.25]  # one retry, first-attempt backoff delay


def test_no_sleep_when_delay_is_zero() -> None:
    slept: list[float] = []
    bad = make_response({"value": "x", "confidence": {"score": 0.8}})
    client = ScriptedClient([bad, make_response(valid_input())])

    DemoRunner(client, RetryPolicy(base_delay_seconds=0.0), sleep=slept.append).run(_ctx())
    assert slept == []


# --------------------------------------------------------------------------- #
# Tool spec generation
# --------------------------------------------------------------------------- #
def test_tool_spec_is_generated_from_contract_schema() -> None:
    spec = DemoRunner(ScriptedClient([])).tool_spec()
    assert spec.name == "emit_demo"
    assert spec.input_schema["type"] == "object"
    assert "value" in spec.input_schema["properties"]
