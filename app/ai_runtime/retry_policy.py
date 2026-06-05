"""Deterministic retry policy for stage execution.

A stage attempt either succeeds or fails with an :class:`~app.ai_runtime.errors.ErrorCode`.
:class:`RetryPolicy` decides, purely as a function of (attempt number, error code),
whether to try again and how long to wait. There is no randomness: no jitter, no
wall-clock dependence. The same inputs always yield the same
:class:`RetryDecision`, which is what lets the research harness replay a run and get
identical behavior.

Backoff is exponential and capped: ``delay(n) = min(max_delay, base * mult**(n-1))``,
where ``n`` is the 1-based number of the attempt that just failed.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.ai_runtime.errors import ErrorCode

__all__ = ["RetryDecision", "RetryPolicy", "DEFAULT_RETRYABLE_CODES"]


#: Failures that are, by default, worth retrying. A retry sends the model
#: deterministic feedback, so these are exactly the codes a corrected re-prompt can
#: plausibly fix. ``CLIENT_ERROR`` is included because transport errors are often
#: transient.
DEFAULT_RETRYABLE_CODES: tuple[ErrorCode, ...] = (
    ErrorCode.TOOL_NOT_CALLED,
    ErrorCode.MALFORMED_TOOL_INPUT,
    ErrorCode.SCHEMA_VALIDATION_FAILED,
    ErrorCode.SEMANTIC_VALIDATION_FAILED,
    ErrorCode.TRUNCATED_OUTPUT,
    ErrorCode.CLIENT_ERROR,
)


class RetryDecision(BaseModel):
    """The outcome of consulting a :class:`RetryPolicy` for one failed attempt."""

    model_config = ConfigDict(frozen=True)

    should_retry: bool = Field(..., description="Whether another attempt should be made.")
    delay_seconds: float = Field(..., ge=0.0, description="How long to wait before retrying.")
    attempt: int = Field(..., ge=1, description="The 1-based attempt that just failed.")
    exhausted: bool = Field(
        default=False,
        description="True when retrying was refused specifically because max_attempts was reached.",
    )
    reason: str = Field(..., min_length=1, description="Human-readable rationale for the decision.")


class RetryPolicy(BaseModel):
    """Deterministic, configurable retry/backoff policy.

    Parameters
    ----------
    max_attempts:
        Total attempts allowed, including the first. ``3`` means one initial try plus
        up to two retries.
    retryable_codes:
        Error codes eligible for retry. Anything else stops immediately.
    base_delay_seconds, backoff_multiplier, max_delay_seconds:
        Exponential-backoff parameters. Defaults give zero delay (test-friendly);
        production callers set a real base.
    """

    model_config = ConfigDict(frozen=True)

    max_attempts: int = Field(default=3, ge=1)
    retryable_codes: tuple[ErrorCode, ...] = Field(default=DEFAULT_RETRYABLE_CODES)
    base_delay_seconds: float = Field(default=0.0, ge=0.0)
    backoff_multiplier: float = Field(default=2.0, ge=1.0)
    max_delay_seconds: float = Field(default=30.0, ge=0.0)

    def is_retryable(self, code: ErrorCode) -> bool:
        """Whether ``code`` is in the configured retryable set."""

        return code in self.retryable_codes

    def compute_delay(self, attempt: int) -> float:
        """Backoff delay (seconds) after a failed ``attempt`` (1-based)."""

        if attempt < 1:
            raise ValueError(f"attempt must be >= 1, got {attempt}")
        raw = self.base_delay_seconds * (self.backoff_multiplier ** (attempt - 1))
        return min(self.max_delay_seconds, raw)

    def decide(self, *, attempt: int, failure_code: ErrorCode | None) -> RetryDecision:
        """Decide whether to retry after ``attempt`` failed with ``failure_code``.

        Parameters
        ----------
        attempt:
            The 1-based number of the attempt that just completed.
        failure_code:
            The error code of that attempt, or ``None`` if it succeeded.

        Returns
        -------
        RetryDecision
            ``should_retry`` plus the backoff delay and a rationale. ``exhausted``
            is set only when a retryable failure was refused purely because
            ``max_attempts`` was reached -- this lets the caller distinguish
            "gave up after trying" from "stopped on a hopeless error".
        """

        if attempt < 1:
            raise ValueError(f"attempt must be >= 1, got {attempt}")

        if failure_code is None:
            return RetryDecision(
                should_retry=False,
                delay_seconds=0.0,
                attempt=attempt,
                reason="attempt succeeded; no retry needed",
            )

        if not self.is_retryable(failure_code):
            return RetryDecision(
                should_retry=False,
                delay_seconds=0.0,
                attempt=attempt,
                reason=f"error {failure_code.value!r} is not retryable",
            )

        if attempt >= self.max_attempts:
            # "Exhausted" means a retry budget existed and was used up. With a
            # single permitted attempt there was nothing to exhaust, so the caller
            # gets the failure's kind instead (via StageStatus.for_error).
            return RetryDecision(
                should_retry=False,
                delay_seconds=0.0,
                attempt=attempt,
                exhausted=self.max_attempts > 1,
                reason=f"max_attempts ({self.max_attempts}) reached for retryable error "
                f"{failure_code.value!r}",
            )

        delay = self.compute_delay(attempt)
        return RetryDecision(
            should_retry=True,
            delay_seconds=delay,
            attempt=attempt,
            reason=f"retrying after retryable error {failure_code.value!r} "
            f"(attempt {attempt}/{self.max_attempts}, delay {delay:.3f}s)",
        )
