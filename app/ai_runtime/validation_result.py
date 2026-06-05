"""Structured validation results and deterministic retry feedback.

Both kinds of validation in the harness -- schema (Pydantic) and semantic
(stage-specific gates like grounding) -- report through one type,
:class:`ValidationResult`. A result is a list of :class:`ValidationIssue` objects;
the stage *fails* iff any issue is an ``ERROR`` (``WARNING`` issues are recorded but
do not block, e.g. a soft "JTBD phrasing looks off" note).

The class also owns :meth:`ValidationResult.retry_feedback`, which turns the error
issues into a single deterministic correction message. Determinism here is
essential: the same set of issues always yields byte-identical feedback, so the
retry loop is reproducible and the research harness can replay it exactly.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

__all__ = [
    "Severity",
    "ValidationIssue",
    "ValidationResult",
]


class Severity(str, Enum):
    """Whether an issue blocks the stage (``ERROR``) or is advisory (``WARNING``)."""

    ERROR = "error"
    WARNING = "warning"


class ValidationIssue(BaseModel):
    """One problem found in a contract.

    Attributes
    ----------
    code:
        Stable machine code, e.g. ``"schema.greater_than"`` or
        ``"grounding.ungrounded_claim"``. Used for metrics and assertions.
    message:
        Human-readable description of what is wrong.
    field:
        Dotted path to the offending field, when applicable
        (e.g. ``"extracted_claims.0.source_span"``).
    severity:
        :class:`Severity`. Only ``ERROR`` issues fail the stage.
    hint:
        Optional, actionable instruction echoed back to the model on retry
        (e.g. "source_span must lie within the provided raw_text").
    context:
        Structured extras (rejected value, threshold, etc.).
    """

    model_config = ConfigDict(frozen=True)

    code: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    field: str | None = None
    severity: Severity = Severity.ERROR
    hint: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)


class ValidationResult(BaseModel):
    """Immutable collection of validation issues with combinator helpers.

    Construct via the classmethods (:meth:`success`, :meth:`failure`,
    :meth:`single_error`) or :meth:`from_pydantic_error`. Combine with
    :meth:`merge`. Inspect with :attr:`ok`, :attr:`errors`, :attr:`warnings`.
    """

    model_config = ConfigDict(frozen=True)

    issues: tuple[ValidationIssue, ...] = ()

    # ----------------------------------------------------------------- builders
    @classmethod
    def success(cls) -> "ValidationResult":
        """A passing result with no issues."""

        return cls(issues=())

    @classmethod
    def failure(cls, issues: Iterable[ValidationIssue]) -> "ValidationResult":
        """A result wrapping the given issues."""

        return cls(issues=tuple(issues))

    @classmethod
    def single_error(
        cls,
        *,
        code: str,
        message: str,
        field: str | None = None,
        hint: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> "ValidationResult":
        """Convenience for a result containing exactly one ``ERROR`` issue."""

        return cls.failure(
            [
                ValidationIssue(
                    code=code,
                    message=message,
                    field=field,
                    severity=Severity.ERROR,
                    hint=hint,
                    context=context or {},
                )
            ]
        )

    @classmethod
    def from_pydantic_error(cls, exc: ValidationError) -> "ValidationResult":
        """Translate a Pydantic :class:`ValidationError` into issues.

        Each underlying error becomes one ``ERROR`` issue whose ``code`` is
        ``"schema.<error_type>"`` and whose ``field`` is the dotted location. The
        offending input is preserved in ``context`` so a retry message can be
        specific.
        """

        issues: list[ValidationIssue] = []
        for err in exc.errors():
            location = ".".join(str(part) for part in err.get("loc", ()))
            issues.append(
                ValidationIssue(
                    code=f"schema.{err.get('type', 'value_error')}",
                    message=str(err.get("msg", "invalid value")),
                    field=location or None,
                    severity=Severity.ERROR,
                    context={"input": err.get("input")},
                )
            )
        if not issues:  # defensive: never produce an empty failure
            issues.append(
                ValidationIssue(
                    code="schema.unknown",
                    message="schema validation failed",
                    severity=Severity.ERROR,
                )
            )
        return cls.failure(issues)

    # --------------------------------------------------------------- combinators
    def merge(self, other: "ValidationResult") -> "ValidationResult":
        """Return a new result containing this result's issues followed by ``other``'s."""

        return ValidationResult(issues=self.issues + other.issues)

    # ----------------------------------------------------------------- accessors
    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        """Only the blocking (``ERROR``) issues."""

        return tuple(i for i in self.issues if i.severity is Severity.ERROR)

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        """Only the advisory (``WARNING``) issues."""

        return tuple(i for i in self.issues if i.severity is Severity.WARNING)

    @property
    def ok(self) -> bool:
        """``True`` iff there are no ``ERROR`` issues (warnings are allowed)."""

        return len(self.errors) == 0

    # ------------------------------------------------------------- retry message
    def retry_feedback(self) -> str:
        """Build a deterministic correction message from the ``ERROR`` issues.

        The output is stable for a given ordered set of issues: same issues in,
        same string out. Returns an empty string when there is nothing to correct.
        """

        errors = self.errors
        if not errors:
            return ""
        lines = [
            "Your previous tool output was rejected. Fix every issue below and call "
            "the tool again with corrected input:"
        ]
        for index, issue in enumerate(errors, start=1):
            location = f" (field: {issue.field})" if issue.field else ""
            hint = f" Hint: {issue.hint}" if issue.hint else ""
            lines.append(f"{index}. [{issue.code}]{location} {issue.message}.{hint}")
        return "\n".join(lines)
