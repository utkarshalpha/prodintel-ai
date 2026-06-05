"""Provider-neutral interfaces and DTOs for the AI execution harness.

The harness never imports the Anthropic SDK directly. Instead it talks to a
:class:`ToolCallClient` whose request/response shapes are defined here. A concrete
Claude adapter implements this protocol in the application layer; tests implement a
scripted fake. This single seam is what makes the entire harness runnable with no
network access and no API key.

All DTOs are immutable Pydantic models so a stage's inputs cannot be mutated
mid-run.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from app.ai_runtime.validation_result import ValidationResult

__all__ = [
    "ToolSpec",
    "LLMMessage",
    "LLMToolResponse",
    "StageContext",
    "ToolCallClient",
    "StageValidator",
]


class ToolSpec(BaseModel):
    """A tool definition handed to the model, mirroring Anthropic's tool shape.

    ``input_schema`` is produced from a contract via ``model_json_schema()`` so the
    tool the model sees and the validator the harness runs are generated from one
    definition and cannot disagree.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    input_schema: dict[str, Any] = Field(...)


class LLMMessage(BaseModel):
    """One conversation turn, in provider-neutral form.

    The concrete client is responsible for translating these into Anthropic content
    blocks (including any tool-use/tool-result framing). Keeping the harness's view
    simple keeps the retry loop easy to reason about and test.
    """

    model_config = ConfigDict(frozen=True)

    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1)


class LLMToolResponse(BaseModel):
    """Normalized result of a single tool-forced model call.

    Attributes
    ----------
    tool_name:
        Name of the tool the model invoked, or ``None`` if it called no tool.
    tool_input:
        The raw JSON object the model passed to the tool, or ``None``.
    stop_reason:
        Provider stop reason (e.g. ``"tool_use"``, ``"max_tokens"``).
    model_id, input_tokens, output_tokens:
        Provenance/metrics for this call.
    text:
        Any free-text preamble the model produced alongside the tool call.
    """

    model_config = ConfigDict(frozen=True)

    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    stop_reason: str = Field(..., min_length=1)
    model_id: str = Field(..., min_length=1)
    input_tokens: int = Field(..., ge=0)
    output_tokens: int = Field(..., ge=0)
    text: str | None = None


class StageContext(BaseModel):
    """Base class for the per-stage inputs a runner needs.

    Each stage subclasses this to declare its own fields (Stage 1 adds the signal's
    ``raw_text``; Stage 4 adds scores, conflicts and retrieved chunks). The base is
    frozen so inputs are stable across retries.
    """

    model_config = ConfigDict(frozen=True)

    stage_name: str = Field(..., min_length=1, description="Identifier of the stage being run.")


@runtime_checkable
class ToolCallClient(Protocol):
    """The single boundary between the harness and a real LLM provider."""

    def complete(
        self,
        *,
        system: str,
        messages: Sequence[LLMMessage],
        tool: ToolSpec,
    ) -> LLMToolResponse:
        """Make one tool-forced call. Must raise
        :class:`~app.ai_runtime.errors.LLMClientError` on transport/API failure."""
        ...


@runtime_checkable
class StageValidator(Protocol):
    """A stage-specific semantic gate run after schema validation succeeds.

    Implementations are pure and deterministic: given the validated contract and
    the stage context, return a :class:`ValidationResult`. Grounding (Stage 1) and
    evidence-completeness (Stage 4) are implemented as validators.
    """

    def validate(self, contract: BaseModel, context: StageContext) -> ValidationResult:
        ...
