"""Error taxonomy for the AI execution harness.

Every failure inside a stage run is reduced to a single :class:`ErrorCode`. That
code drives two decisions deterministically:

* whether the failure is *retryable* (see :mod:`app.ai_runtime.retry_policy`), and
* which terminal :class:`~app.ai_runtime.stage_result.StageStatus` the run ends in.

Keeping the taxonomy small and explicit is what makes the retry loop predictable:
there are no "unknown" branches, every code has a defined disposition.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ErrorCode",
    "StageError",
    "LLMClientError",
    "StageExecutionError",
]


class ErrorCode(str, Enum):
    """Closed set of failure categories a stage attempt can produce."""

    #: The model did not call the required tool (or called the wrong one).
    TOOL_NOT_CALLED = "tool_not_called"
    #: The tool was called but its input was not a usable JSON object.
    MALFORMED_TOOL_INPUT = "malformed_tool_input"
    #: The tool input failed Pydantic schema validation.
    SCHEMA_VALIDATION_FAILED = "schema_validation_failed"
    #: The contract was schema-valid but failed a stage-specific semantic gate.
    SEMANTIC_VALIDATION_FAILED = "semantic_validation_failed"
    #: The model's output was truncated (e.g. stop_reason == "max_tokens").
    TRUNCATED_OUTPUT = "truncated_output"
    #: The underlying client call raised (network/API error).
    CLIENT_ERROR = "client_error"


class StageError(BaseModel):
    """A single, structured failure from one stage attempt.

    Carries enough context to (a) drive the retry decision, (b) be surfaced to a
    PM, and (c) be recorded in metrics. ``recoverable`` is the model's own opinion
    of transience; the :class:`~app.ai_runtime.retry_policy.RetryPolicy` has the
    final say on whether a retry actually happens.
    """

    model_config = ConfigDict(frozen=True)

    code: ErrorCode = Field(..., description="Category of the failure.")
    message: str = Field(..., description="Human-readable explanation.")
    attempt: int = Field(..., ge=1, description="1-based attempt number that produced this error.")
    recoverable: bool = Field(
        default=True,
        description="Whether this error is, in principle, worth retrying.",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured context (e.g. rejected fields, stop_reason).",
    )


class LLMClientError(Exception):
    """Raised by a :class:`~app.ai_runtime.interfaces.ToolCallClient` on a transport
    or API failure. The harness catches this and converts it into a
    :class:`StageError` with :attr:`ErrorCode.CLIENT_ERROR`."""


class StageExecutionError(Exception):
    """Raised by :meth:`~app.ai_runtime.stage_result.StageResult.output_or_raise`
    when a caller demands the output of a failed stage run."""

    def __init__(self, error: StageError | None, stage_name: str) -> None:
        self.error = error
        self.stage_name = stage_name
        detail = error.message if error else "unknown error"
        super().__init__(f"stage {stage_name!r} did not produce a valid output: {detail}")
