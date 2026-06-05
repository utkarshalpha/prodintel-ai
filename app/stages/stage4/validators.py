"""Stage 4 semantic validator: decision integrity.

Adapts the deterministic decision-integrity gate
(:func:`app.ai_contracts.validation.decision_integrity.validate_decision_integrity`)
to the harness :class:`~app.ai_runtime.interfaces.StageValidator` protocol. No logic
is reimplemented; this turns the integrity report into harness validation issues
with deterministic retry hints.

Policy: any integrity problem fails the whole synthesis and triggers a retry, so a
persisted decision set is always evidence-traceable and honestly accounts for every
conflict over each subject feature.
"""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence, runtime_checkable
from uuid import UUID

from app.ai_contracts.stage4_decision import DecisionSynthesisContract
from app.ai_contracts.validation.decision_integrity import validate_decision_integrity
from app.ai_runtime.validation_result import Severity, ValidationIssue, ValidationResult

__all__ = ["DecisionIntegrityValidator", "DECISION_ISSUE_CODES"]

DECISION_ISSUE_CODES = {
    "unknown_subject": "decision.unknown_subject",
    "duplicate_subject": "decision.duplicate_subject",
    "unknown_evidence": "decision.unknown_evidence",
    "unknown_conflict": "decision.unknown_conflict",
    "conflict_subject_mismatch": "decision.conflict_subject_mismatch",
    "unacknowledged_conflict": "decision.unacknowledged_conflict",
}

_HINT = (
    "Use only the provided feature_ids as subjects and signal_ids as evidence; decide "
    "each feature at most once; acknowledge every conflict_id whose subject_id matches "
    "the feature you are deciding, and acknowledge no others."
)


@runtime_checkable
class SupportsDecisionInputs(Protocol):
    """Structural type for a context exposing the input feature/signal/conflict sets."""

    @property
    def input_feature_ids(self) -> Sequence[UUID]: ...

    @property
    def input_signal_ids(self) -> Sequence[UUID]: ...

    @property
    def conflict_subjects(self) -> Mapping[UUID, UUID]: ...


class DecisionIntegrityValidator:
    """Verify subject validity, evidence traceability, and conflict accounting."""

    def validate(
        self,
        contract: DecisionSynthesisContract,
        context: SupportsDecisionInputs,
    ) -> ValidationResult:
        report = validate_decision_integrity(
            contract,
            context.input_feature_ids,
            context.input_signal_ids,
            context.conflict_subjects,
        )
        if report.passed:
            return ValidationResult.success()

        issues: list[ValidationIssue] = []

        for index, subject_id in report.unknown_subjects:
            issues.append(
                _error(
                    DECISION_ISSUE_CODES["unknown_subject"],
                    f"decision #{index} references subject {subject_id} which is not an input feature",
                    field=f"decisions.{index}.subject_id",
                )
            )
        for index in report.duplicate_subjects:
            issues.append(
                _error(
                    DECISION_ISSUE_CODES["duplicate_subject"],
                    f"decision #{index} decides a feature that another decision already decides",
                    field=f"decisions.{index}.subject_id",
                )
            )
        for index, signal_id in report.unknown_evidence:
            issues.append(
                _error(
                    DECISION_ISSUE_CODES["unknown_evidence"],
                    f"decision #{index} cites evidence signal {signal_id} which is not in the input set",
                    field=f"decisions.{index}.evidence_signal_ids",
                )
            )
        for index, conflict_id in report.unknown_conflicts:
            issues.append(
                _error(
                    DECISION_ISSUE_CODES["unknown_conflict"],
                    f"decision #{index} acknowledges conflict {conflict_id} which is not in the input set",
                    field=f"decisions.{index}.acknowledged_conflict_ids",
                )
            )
        for index, conflict_id in report.conflict_subject_mismatch:
            issues.append(
                _error(
                    DECISION_ISSUE_CODES["conflict_subject_mismatch"],
                    f"decision #{index} acknowledges conflict {conflict_id} which is about a different feature",
                    field=f"decisions.{index}.acknowledged_conflict_ids",
                )
            )
        for index, conflict_id in report.unacknowledged_conflicts:
            issues.append(
                _error(
                    DECISION_ISSUE_CODES["unacknowledged_conflict"],
                    f"decision #{index} ignores conflict {conflict_id} which was detected over its subject feature",
                    field=f"decisions.{index}.acknowledged_conflict_ids",
                )
            )

        return ValidationResult.failure(issues)


def _error(code: str, message: str, *, field: str | None = None) -> ValidationIssue:
    return ValidationIssue(code=code, message=message, field=field, severity=Severity.ERROR, hint=_HINT)
