"""Metrics collected during stage execution.

Two levels: one :class:`AttemptMetrics` per model call, and one
:class:`StageMetrics` aggregating the whole run. Token counts and per-attempt
outcomes feed both operational dashboards and the research paper's cost/quality
analysis, so they are first-class, not logging side effects.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, computed_field

from app.ai_runtime.errors import ErrorCode

__all__ = ["AttemptMetrics", "StageMetrics"]


class AttemptMetrics(BaseModel):
    """Metrics for a single model call within a stage run."""

    model_config = ConfigDict(frozen=True)

    attempt: int = Field(..., ge=1)
    model_id: str = Field(..., min_length=1)
    input_tokens: int = Field(..., ge=0)
    output_tokens: int = Field(..., ge=0)
    stop_reason: str = Field(..., min_length=1)
    schema_valid: bool = Field(..., description="Did the tool input pass schema validation?")
    semantic_valid: bool = Field(..., description="Did the contract pass semantic validation?")
    error_code: ErrorCode | None = Field(default=None, description="Failure code, if this attempt failed.")
    latency_ms: float | None = Field(default=None, ge=0.0, description="Wall-clock latency, if measured.")


class StageMetrics(BaseModel):
    """Aggregate metrics for an entire stage run.

    Built incrementally by the runner: one :class:`AttemptMetrics` is appended per
    call, and ``succeeded``/``final_error_code`` are stamped at the end. Token
    totals are computed properties so they cannot drift from the attempt list.
    """

    model_config = ConfigDict(frozen=True)

    stage_name: str = Field(..., min_length=1)
    attempts: tuple[AttemptMetrics, ...] = ()
    succeeded: bool = False
    final_error_code: ErrorCode | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_attempts(self) -> int:
        """Number of model calls made."""

        return len(self.attempts)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_input_tokens(self) -> int:
        """Sum of input tokens across all attempts."""

        return sum(a.input_tokens for a in self.attempts)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_output_tokens(self) -> int:
        """Sum of output tokens across all attempts."""

        return sum(a.output_tokens for a in self.attempts)

    def with_attempt(self, attempt: AttemptMetrics) -> "StageMetrics":
        """Return a new :class:`StageMetrics` with ``attempt`` appended."""

        return self.model_copy(update={"attempts": self.attempts + (attempt,)})

    def finalize(self, *, succeeded: bool, final_error_code: ErrorCode | None) -> "StageMetrics":
        """Return a new :class:`StageMetrics` with terminal outcome stamped."""

        return self.model_copy(update={"succeeded": succeeded, "final_error_code": final_error_code})
