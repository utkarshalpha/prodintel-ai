"""Integration tests for Stage1SignalRunner -- the first production stage.

Exercises the full validation flow end-to-end with a scripted client:
Signal -> tool call -> ParsedSignalContract -> GroundingValidator -> retry -> result.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from app.ai_contracts.enums import StakeholderType
from app.ai_runtime.errors import ErrorCode
from app.ai_runtime.retry_policy import RetryPolicy
from app.ai_runtime.stage_result import StageStatus
from app.stages.stage1 import prompts
from app.stages.stage1.runner import Stage1Context, build_stage1_runner
from app.stages.stage1.validators import GROUNDING_ISSUE_CODE

from tests.runtime_helpers import ScriptedClient, make_response

RAW = "The mobile checkout fails when the user taps pay."
GROUNDED_CLAIM = {"text": "mobile checkout fails", "source_span": [4, 25], "claim_confidence": 0.9}
UNGROUNDED_CLAIM = {"text": "a refund was requested", "source_span": [0, 3], "claim_confidence": 0.9}


def _payload(signal_id: UUID, claims: list[dict[str, Any]], *, urgency: int = 4, score: float = 0.82) -> dict[str, Any]:
    """A *semantic-only* tool input (no model_meta) -- what the model returns given
    the stripped Stage 1 tool schema."""

    return {
        "signal_id": str(signal_id),
        "intent": "Fix mobile checkout failures",
        "stakeholder_type": "customer",
        "urgency": urgency,
        "sentiment": -0.3,
        "extracted_claims": claims,
        "confidence": {"score": score, "components": {"span_grounding_ratio": 1.0}},
    }


def _response(signal_id: UUID, claims: list[dict[str, Any]], **kwargs: Any):
    return make_response(_payload(signal_id, claims, **kwargs), tool_name=prompts.STAGE1_TOOL_NAME)


def _ctx(signal_id: UUID) -> Stage1Context:
    return Stage1Context(signal_id=signal_id, raw_text=RAW, source_type=StakeholderType.CUSTOMER)


# --------------------------------------------------------------------------- #
# Happy path + harness-managed metadata injection
# --------------------------------------------------------------------------- #
def test_success_first_attempt() -> None:
    sid = uuid4()
    client = ScriptedClient([_response(sid, [GROUNDED_CLAIM])])
    result = build_stage1_runner(client).run(_ctx(sid))

    assert result.succeeded is True
    assert result.status is StageStatus.SUCCESS
    assert result.output is not None
    assert result.output.signal_id == sid
    assert result.attempts_used == 1
    assert result.confidence is not None and result.confidence.score == 0.82


def test_model_meta_is_injected_from_response_not_the_model() -> None:
    sid = uuid4()
    client = ScriptedClient([_response(sid, [GROUNDED_CLAIM])])
    result = build_stage1_runner(client).run(_ctx(sid))

    meta = result.output_or_raise().model_meta
    assert meta.model_id == "claude-test"        # from the response
    assert meta.input_tokens == 100              # from the response, not fabricated
    assert meta.output_tokens == 20
    assert meta.prompt_version == prompts.STAGE1_PROMPT_VERSION
    assert meta.stop_reason == "tool_use"


# --------------------------------------------------------------------------- #
# Grounding retry flow
# --------------------------------------------------------------------------- #
def test_ungrounded_then_grounded_retries_and_succeeds() -> None:
    sid = uuid4()
    client = ScriptedClient(
        [
            _response(sid, [UNGROUNDED_CLAIM]),  # grounding fails
            _response(sid, [GROUNDED_CLAIM]),    # grounding passes
        ]
    )
    result = build_stage1_runner(client).run(_ctx(sid))

    assert result.succeeded is True
    assert result.attempts_used == 2
    # First attempt: schema-valid but semantically rejected by grounding.
    first = result.metrics.attempts[0]
    assert first.schema_valid is True
    assert first.semantic_valid is False
    assert first.error_code is ErrorCode.SEMANTIC_VALIDATION_FAILED
    # The retry carried deterministic grounding feedback to the model.
    correction = "\n".join(m.content for m in client.received[1])
    assert "not grounded" in correction
    assert GROUNDING_ISSUE_CODE in correction


def test_grounding_failure_exhausts() -> None:
    sid = uuid4()
    client = ScriptedClient([_response(sid, [UNGROUNDED_CLAIM]), _response(sid, [UNGROUNDED_CLAIM])])
    result = build_stage1_runner(client, retry_policy=RetryPolicy(max_attempts=2)).run(_ctx(sid))

    assert result.succeeded is False
    assert result.status is StageStatus.FAILED_EXHAUSTED
    assert result.error is not None
    assert result.error.code is ErrorCode.SEMANTIC_VALIDATION_FAILED


# --------------------------------------------------------------------------- #
# Schema retry flow
# --------------------------------------------------------------------------- #
def test_schema_failure_then_success() -> None:
    sid = uuid4()
    client = ScriptedClient(
        [
            _response(sid, [GROUNDED_CLAIM], urgency=9),  # urgency out of range -> schema fail
            _response(sid, [GROUNDED_CLAIM]),
        ]
    )
    result = build_stage1_runner(client).run(_ctx(sid))

    assert result.succeeded is True
    assert result.attempts_used == 2
    assert result.metrics.attempts[0].schema_valid is False
    assert result.metrics.attempts[0].error_code is ErrorCode.SCHEMA_VALIDATION_FAILED


# --------------------------------------------------------------------------- #
# Tool spec: harness-managed fields are hidden from the model
# --------------------------------------------------------------------------- #
def test_tool_spec_strips_harness_managed_fields() -> None:
    spec = build_stage1_runner(ScriptedClient([])).tool_spec()
    props = spec.input_schema["properties"]
    required = spec.input_schema["required"]

    assert spec.name == prompts.STAGE1_TOOL_NAME
    assert "model_meta" not in props
    assert "schema_version" not in props
    assert "model_meta" not in required
    # Semantic fields the model must provide are present.
    assert "signal_id" in props
    assert "extracted_claims" in props


# --------------------------------------------------------------------------- #
# Prompt package
# --------------------------------------------------------------------------- #
def test_prompts_contain_required_anchors() -> None:
    sid = uuid4()
    system = prompts.build_system_prompt()
    user = prompts.build_user_prompt(signal_id=sid, raw_text=RAW, source_type=StakeholderType.SALES)

    assert prompts.STAGE1_TOOL_NAME in system
    assert str(sid) in user
    assert RAW in user
    assert "<<<SIGNAL>>>" in user
    assert "sales" in user  # source-channel hint rendered
