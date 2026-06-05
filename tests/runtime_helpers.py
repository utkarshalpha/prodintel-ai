"""Reusable test doubles and fixtures for exercising the AI execution harness.

Defines a tiny self-contained stage (``DemoContract`` / ``DemoContext`` /
``DemoRunner``) plus a ``ScriptedClient`` so the harness can be tested with no
network, no API key, and fully deterministic model responses.
"""

from __future__ import annotations

from typing import Any, Sequence

from pydantic import Field

from app.ai_contracts.base import ConfidenceBlock, FrozenModel
from app.ai_runtime.base_runner import BaseStageRunner
from app.ai_runtime.errors import LLMClientError
from app.ai_runtime.interfaces import (
    LLMMessage,
    LLMToolResponse,
    StageContext,
    StageValidator,
    ToolSpec,
)
from app.ai_runtime.validation_result import ValidationResult

TOOL_NAME = "emit_demo"


class DemoContract(FrozenModel):
    """Minimal contract: an integer plus a confidence block (for collection tests)."""

    value: int = Field(..., ge=0)
    confidence: ConfidenceBlock


class DemoContext(StageContext):
    """Context carrying a minimum the demo semantic validator enforces."""

    min_value: int = 0


class MinValueValidator:
    """Semantic gate: ``contract.value`` must be >= ``context.min_value``."""

    def validate(self, contract: Any, context: Any) -> ValidationResult:
        if contract.value < context.min_value:
            return ValidationResult.single_error(
                code="demo.below_min",
                message=f"value {contract.value} is below required minimum {context.min_value}",
                field="value",
                hint=f"return a value of at least {context.min_value}",
            )
        return ValidationResult.success()


class DemoRunner(BaseStageRunner[DemoContract, DemoContext]):
    """Concrete runner wiring the demo contract into the harness."""

    @property
    def stage_name(self) -> str:
        return "demo"

    @property
    def tool_name(self) -> str:
        return TOOL_NAME

    @property
    def tool_description(self) -> str:
        return "Emit a demo contract."

    def contract_model(self) -> type[DemoContract]:
        return DemoContract

    def build_system_prompt(self, context: DemoContext) -> str:
        return "You are a demo stage."

    def build_user_prompt(self, context: DemoContext) -> str:
        return f"Emit a value of at least {context.min_value}."

    def semantic_validators(self) -> Sequence[StageValidator]:
        return (MinValueValidator(),)


class ScriptedClient:
    """A ``ToolCallClient`` that replays a fixed script of responses/exceptions.

    Records the messages passed on each call so tests can assert that deterministic
    correction feedback was appended between attempts.
    """

    def __init__(self, script: Sequence[LLMToolResponse | Exception]) -> None:
        self._script: list[LLMToolResponse | Exception] = list(script)
        self.received: list[list[LLMMessage]] = []
        self.call_count = 0

    def complete(self, *, system: str, messages: Sequence[LLMMessage], tool: ToolSpec) -> LLMToolResponse:
        self.received.append(list(messages))
        self.call_count += 1
        if not self._script:
            raise AssertionError("ScriptedClient ran out of scripted responses")
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_response(
    tool_input: dict[str, Any] | None,
    *,
    tool_name: str | None = TOOL_NAME,
    stop_reason: str = "tool_use",
    input_tokens: int = 100,
    output_tokens: int = 20,
    model_id: str = "claude-test",
) -> LLMToolResponse:
    """Build a normalized model response for the scripted client."""

    return LLMToolResponse(
        tool_name=tool_name,
        tool_input=tool_input,
        stop_reason=stop_reason,
        model_id=model_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def valid_input(value: int = 10, score: float = 0.8) -> dict[str, Any]:
    """A tool input that passes DemoContract schema validation."""

    return {"value": value, "confidence": {"score": score}}


def client_error(message: str = "boom") -> LLMClientError:
    """An exception the scripted client will raise to simulate a transport failure."""

    return LLMClientError(message)
