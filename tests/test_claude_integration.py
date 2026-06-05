"""Integration tests: ClaudeToolClient driving the real Stage 1 runner via mocks.

Wires the adapter (backed by a fake Anthropic SDK) into the actual
``Stage1SignalRunner`` and runs the full validation flow, confirming the adapter's
``LLMToolResponse`` is compatible with the existing harness end-to-end.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from app.ai_clients.claude_client import ClaudeToolClient
from app.ai_clients.config import ClaudeClientConfig
from app.ai_runtime.errors import ErrorCode
from app.ai_runtime.retry_policy import RetryPolicy
from app.stages.stage1 import prompts
from app.stages.stage1.runner import Stage1Context, build_stage1_runner

from tests.anthropic_fakes import FakeAnthropic, rate_limit_error, tool_message

RAW = "The mobile checkout fails when the user taps pay."


def _config() -> ClaudeClientConfig:
    return ClaudeClientConfig(api_key="sk-test", model="claude-test", max_tokens=512, timeout_seconds=5)


def _stage1_payload(signal_id, raw: str) -> dict[str, Any]:
    """Semantic-only payload (no model_meta) -- what the model returns under the
    Stage 1 tool schema. A whole-text span is trivially grounded."""

    return {
        "signal_id": str(signal_id),
        "intent": "Fix mobile checkout failures",
        "stakeholder_type": "customer",
        "urgency": 4,
        "sentiment": -0.3,
        "extracted_claims": [{"text": raw, "source_span": [0, len(raw)], "claim_confidence": 0.95}],
        "confidence": {"score": 0.8, "components": {"span_grounding_ratio": 1.0}},
    }


def _runner(behaviors, *, retry_policy: RetryPolicy | None = None):
    client = ClaudeToolClient(_config(), client=FakeAnthropic(behaviors))
    return build_stage1_runner(client, retry_policy=retry_policy)


def test_adapter_drives_stage1_to_success() -> None:
    sid = uuid4()
    message = tool_message(prompts.STAGE1_TOOL_NAME, _stage1_payload(sid, RAW), input_tokens=130, output_tokens=40)
    runner = _runner([message])

    result = runner.run(Stage1Context(signal_id=sid, raw_text=RAW))

    assert result.succeeded is True
    assert result.output is not None
    assert result.output.signal_id == sid
    # model_meta is harness-injected from the adapter's response.
    assert result.output.model_meta.model_id == "claude-test"
    assert result.output.model_meta.input_tokens == 130
    assert result.confidence is not None and result.confidence.score == 0.8


def test_adapter_transient_error_is_retried_then_succeeds() -> None:
    sid = uuid4()
    message = tool_message(prompts.STAGE1_TOOL_NAME, _stage1_payload(sid, RAW))
    # First call rate-limited (retryable) -> harness retries -> second call succeeds.
    runner = _runner([rate_limit_error(), message], retry_policy=RetryPolicy(max_attempts=3))

    result = runner.run(Stage1Context(signal_id=sid, raw_text=RAW))

    assert result.succeeded is True
    assert result.attempts_used == 2
    assert result.metrics.attempts[0].error_code is ErrorCode.CLIENT_ERROR
    assert result.metrics.attempts[0].stop_reason == "client_error"
