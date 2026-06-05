"""Unit tests for ValidationResult, ValidationIssue, and deterministic feedback."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field, ValidationError

from app.ai_runtime.validation_result import Severity, ValidationIssue, ValidationResult


def test_success_is_ok_and_empty() -> None:
    result = ValidationResult.success()
    assert result.ok is True
    assert result.issues == ()
    assert result.retry_feedback() == ""


def test_single_error_fails() -> None:
    result = ValidationResult.single_error(code="x.bad", message="nope", field="a", hint="fix a")
    assert result.ok is False
    assert len(result.errors) == 1
    assert result.errors[0].field == "a"


def test_warnings_do_not_fail() -> None:
    warning = ValidationIssue(code="w", message="soft", severity=Severity.WARNING)
    result = ValidationResult.failure([warning])
    assert result.ok is True
    assert result.warnings == (warning,)
    assert result.errors == ()


def test_merge_concatenates_in_order() -> None:
    a = ValidationResult.single_error(code="a", message="ma")
    b = ValidationResult.single_error(code="b", message="mb")
    merged = a.merge(b)
    assert [i.code for i in merged.issues] == ["a", "b"]
    assert merged.ok is False


def test_retry_feedback_is_deterministic() -> None:
    result = ValidationResult.failure(
        [
            ValidationIssue(code="c1", message="m1", field="f1", hint="h1"),
            ValidationIssue(code="c2", message="m2"),
        ]
    )
    first = result.retry_feedback()
    second = result.retry_feedback()
    assert first == second  # same issues -> identical message
    assert "[c1] (field: f1) m1. Hint: h1" in first
    assert "[c2] m2." in first


def test_retry_feedback_ignores_warnings() -> None:
    result = ValidationResult.failure(
        [ValidationIssue(code="w", message="soft", severity=Severity.WARNING)]
    )
    assert result.retry_feedback() == ""


def test_from_pydantic_error_builds_issues() -> None:
    class _M(BaseModel):
        n: int = Field(..., ge=5)

    with pytest.raises(ValidationError) as exc_info:
        _M(n=1)

    result = ValidationResult.from_pydantic_error(exc_info.value)
    assert result.ok is False
    issue = result.errors[0]
    assert issue.field == "n"
    assert issue.code.startswith("schema.")
