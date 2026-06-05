"""Stage 3 semantic validator: conflict integrity.

Adapts the deterministic conflict-integrity gate
(:func:`app.ai_contracts.validation.conflict_integrity.validate_conflict_integrity`)
to the harness :class:`~app.ai_runtime.interfaces.StageValidator` protocol. No logic
is reimplemented; this turns the integrity report into harness validation issues
with deterministic retry hints.

Policy: any integrity problem fails the whole detection and triggers a retry, so a
persisted conflict set is always evidence-traceable and represents real opposition.
An empty conflict list passes (stakeholders agree).
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable
from uuid import UUID

from app.ai_contracts.stage3_conflict import ConflictDetectionContract
from app.ai_contracts.validation.conflict_integrity import validate_conflict_integrity
from app.ai_runtime.validation_result import Severity, ValidationIssue, ValidationResult

__all__ = ["ConflictIntegrityValidator", "CONFLICT_ISSUE_CODES"]

CONFLICT_ISSUE_CODES = {
    "unknown_subject": "conflict.unknown_subject",
    "duplicate_stakeholders": "conflict.duplicate_stakeholders",
    "stakeholder_mismatch": "conflict.stakeholder_mismatch",
    "unknown_evidence": "conflict.unknown_evidence",
    "missing_opposition": "conflict.missing_opposition",
}

_HINT = (
    "Use only the provided feature_ids as subjects and signal_ids as evidence; ensure "
    "each conflict has >= 2 distinct stakeholders with genuinely opposing stances. If "
    "stakeholders agree, return an empty conflicts list."
)


@runtime_checkable
class SupportsConflictInputs(Protocol):
    """Structural type for a context exposing the input feature and signal id sets."""

    @property
    def input_feature_ids(self) -> Sequence[UUID]: ...

    @property
    def input_signal_ids(self) -> Sequence[UUID]: ...


class ConflictIntegrityValidator:
    """Verify subject validity, evidence traceability, and genuine opposition."""

    def validate(
        self,
        contract: ConflictDetectionContract,
        context: SupportsConflictInputs,
    ) -> ValidationResult:
        report = validate_conflict_integrity(
            contract, context.input_feature_ids, context.input_signal_ids
        )
        if report.passed:
            return ValidationResult.success()

        issues: list[ValidationIssue] = []

        for index, subject_id in report.unknown_subjects:
            issues.append(
                _error(
                    CONFLICT_ISSUE_CODES["unknown_subject"],
                    f"conflict #{index} references subject {subject_id} which is not an input feature",
                    field=f"conflicts.{index}.subject_id",
                )
            )
        for index in report.duplicate_stakeholders:
            issues.append(
                _error(
                    CONFLICT_ISSUE_CODES["duplicate_stakeholders"],
                    f"conflict #{index} lists the same stakeholder more than once",
                    field=f"conflicts.{index}.stakeholders",
                )
            )
        for index in report.stakeholder_mismatch:
            issues.append(
                _error(
                    CONFLICT_ISSUE_CODES["stakeholder_mismatch"],
                    f"conflict #{index} stakeholders do not match its positions",
                    field=f"conflicts.{index}.positions",
                )
            )
        for index, signal_id in report.unknown_evidence:
            issues.append(
                _error(
                    CONFLICT_ISSUE_CODES["unknown_evidence"],
                    f"conflict #{index} cites evidence signal {signal_id} which is not in the input set",
                    field=f"conflicts.{index}.positions",
                )
            )
        for index in report.missing_opposition:
            issues.append(
                _error(
                    CONFLICT_ISSUE_CODES["missing_opposition"],
                    f"conflict #{index} has no opposing positions (it is not a genuine conflict)",
                    field=f"conflicts.{index}.positions",
                )
            )

        return ValidationResult.failure(issues)


def _error(code: str, message: str, *, field: str | None = None) -> ValidationIssue:
    return ValidationIssue(code=code, message=message, field=field, severity=Severity.ERROR, hint=_HINT)
