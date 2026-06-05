"""Unit tests for the deterministic retry policy."""

from __future__ import annotations

import pytest

from app.ai_runtime.errors import ErrorCode
from app.ai_runtime.retry_policy import RetryPolicy


def test_success_never_retries() -> None:
    decision = RetryPolicy().decide(attempt=1, failure_code=None)
    assert decision.should_retry is False
    assert decision.exhausted is False


def test_non_retryable_code_stops_immediately() -> None:
    policy = RetryPolicy(retryable_codes=(ErrorCode.SCHEMA_VALIDATION_FAILED,))
    decision = policy.decide(attempt=1, failure_code=ErrorCode.CLIENT_ERROR)
    assert decision.should_retry is False
    assert decision.exhausted is False
    assert "not retryable" in decision.reason


def test_retryable_within_budget_retries() -> None:
    policy = RetryPolicy(max_attempts=3)
    decision = policy.decide(attempt=1, failure_code=ErrorCode.SCHEMA_VALIDATION_FAILED)
    assert decision.should_retry is True
    assert decision.exhausted is False


def test_retryable_at_max_is_exhausted() -> None:
    policy = RetryPolicy(max_attempts=2)
    decision = policy.decide(attempt=2, failure_code=ErrorCode.SCHEMA_VALIDATION_FAILED)
    assert decision.should_retry is False
    assert decision.exhausted is True
    assert "max_attempts" in decision.reason


def test_compute_delay_is_exponential_and_capped() -> None:
    policy = RetryPolicy(base_delay_seconds=1.0, backoff_multiplier=2.0, max_delay_seconds=5.0)
    assert policy.compute_delay(1) == pytest.approx(1.0)
    assert policy.compute_delay(2) == pytest.approx(2.0)
    assert policy.compute_delay(3) == pytest.approx(4.0)
    assert policy.compute_delay(4) == pytest.approx(5.0)  # capped


def test_zero_base_delay_means_no_wait() -> None:
    policy = RetryPolicy(base_delay_seconds=0.0)
    decision = policy.decide(attempt=1, failure_code=ErrorCode.SCHEMA_VALIDATION_FAILED)
    assert decision.delay_seconds == 0.0


def test_is_retryable_reflects_config() -> None:
    policy = RetryPolicy(retryable_codes=(ErrorCode.TOOL_NOT_CALLED,))
    assert policy.is_retryable(ErrorCode.TOOL_NOT_CALLED) is True
    assert policy.is_retryable(ErrorCode.SEMANTIC_VALIDATION_FAILED) is False


def test_decide_rejects_bad_attempt() -> None:
    with pytest.raises(ValueError):
        RetryPolicy().decide(attempt=0, failure_code=None)
