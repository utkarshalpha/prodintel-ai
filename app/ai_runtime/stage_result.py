"""The result object returned by every stage run.

:class:`StageResult` is generic over the contract type ``T`` so callers get a fully
typed ``output``. It bundles everything a caller, an auditor, or the research
harness needs from one run: the terminal status, the validated contract (or
``None``), the final validation issues, the terminal error, full metrics, and the
collected confidence block.

Confidence collection lives here by design: any contract that carries a
``confidence`` field has it surfaced to ``StageResult.confidence`` automatically, so
the harness offers a uniform place to read confidence regardless of stage.
"""

from __future__ import annotations

from enum import Enum
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from app.ai_contracts.base import ConfidenceBlock
from app.ai_runtime.errors import ErrorCode, StageError, StageExecutionError
from app.ai_runtime.metrics import StageMetrics
from app.ai_runtime.validation_result import ValidationResult

__all__ = ["StageStatus", "StageResult"]

TContract = TypeVar("TContract", bound=BaseModel)


class StageStatus(str, Enum):
    """Terminal disposition of a stage run."""

    SUCCESS = "success"
    #: The model never produced a usable tool call (not called, malformed, truncated).
    FAILED_TOOL_CALL = "failed_tool_call"
    #: A schema or semantic gate rejected every produced contract.
    FAILED_VALIDATION = "failed_validation"
    #: The underlying client kept raising.
    FAILED_CLIENT = "failed_client"
    #: A retryable failure was never resolved before max_attempts ran out.
    FAILED_EXHAUSTED = "failed_exhausted"

    @classmethod
    def for_error(cls, code: ErrorCode) -> "StageStatus":
        """Map a terminal :class:`ErrorCode` to its non-exhaustion status.

        Exhaustion (giving up after retries) is decided by the runner, not here;
        this maps the *kind* of the final error.
        """

        if code in (ErrorCode.TOOL_NOT_CALLED, ErrorCode.MALFORMED_TOOL_INPUT, ErrorCode.TRUNCATED_OUTPUT):
            return cls.FAILED_TOOL_CALL
        if code in (ErrorCode.SCHEMA_VALIDATION_FAILED, ErrorCode.SEMANTIC_VALIDATION_FAILED):
            return cls.FAILED_VALIDATION
        return cls.FAILED_CLIENT


class StageResult(BaseModel, Generic[TContract]):
    """Typed, self-describing outcome of one stage run.

    Attributes
    ----------
    stage_name:
        Identifier of the stage that produced this result.
    status:
        :class:`StageStatus` terminal disposition.
    output:
        The validated contract on success, else ``None``.
    validation:
        The final :class:`ValidationResult` (may carry warnings even on success).
    error:
        The terminal :class:`StageError` on failure, else ``None``.
    metrics:
        Aggregated :class:`StageMetrics` for the run.
    confidence:
        The contract's confidence block, if it exposes one.
    attempts_used:
        Number of model calls actually made.
    """

    model_config = ConfigDict(frozen=True)

    stage_name: str = Field(..., min_length=1)
    status: StageStatus
    output: TContract | None = None
    validation: ValidationResult = Field(default_factory=ValidationResult.success)
    error: StageError | None = None
    metrics: StageMetrics
    confidence: ConfidenceBlock | None = None
    attempts_used: int = Field(..., ge=0)

    @property
    def succeeded(self) -> bool:
        """``True`` iff the run produced a valid output."""

        return self.status is StageStatus.SUCCESS

    def output_or_raise(self) -> TContract:
        """Return the validated output, or raise :class:`StageExecutionError`.

        Use at call sites that cannot proceed without a contract; the raised error
        carries the terminal :class:`StageError` for handling upstream.
        """

        if self.output is None or not self.succeeded:
            raise StageExecutionError(self.error, self.stage_name)
        return self.output
