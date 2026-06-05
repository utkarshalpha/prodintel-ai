"""Stage 2 semantic validator: feature provenance.

Adapts the deterministic provenance gate
(:func:`app.ai_contracts.validation.provenance.validate_feature_provenance`) to the
harness :class:`~app.ai_runtime.interfaces.StageValidator` protocol. No logic is
reimplemented; this turns the provenance report into harness validation issues with
deterministic retry hints.

Policy: any provenance problem fails the whole extraction and triggers a retry, so a
persisted feature set always has real, fully-partitioned provenance.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable
from uuid import UUID

from app.ai_contracts.stage2_feature import FeatureExtractionContract
from app.ai_contracts.validation.provenance import validate_feature_provenance
from app.ai_runtime.validation_result import Severity, ValidationIssue, ValidationResult

__all__ = ["FeatureProvenanceValidator", "PROVENANCE_ISSUE_CODES"]

PROVENANCE_ISSUE_CODES = {
    "unknown_signal": "provenance.unknown_signal",
    "duplicate": "provenance.duplicate_assignment",
    "unknown_unassigned": "provenance.unknown_unassigned",
    "both": "provenance.assigned_and_unassigned",
    "missing": "provenance.missing_signal",
}

_HINT = (
    "Use only the signal_ids provided in the prompt. Assign each input signal to "
    "exactly one feature's source_signal_ids or to unassigned_signal_ids."
)


@runtime_checkable
class SupportsInputSignalIds(Protocol):
    """Structural type for any context exposing the input signal id set."""

    @property
    def input_signal_ids(self) -> Sequence[UUID]: ...


class FeatureProvenanceValidator:
    """Verify that an extraction's provenance is real and fully partitions the input."""

    def validate(
        self,
        contract: FeatureExtractionContract,
        context: SupportsInputSignalIds,
    ) -> ValidationResult:
        report = validate_feature_provenance(contract, context.input_signal_ids)
        if report.passed:
            return ValidationResult.success()

        issues: list[ValidationIssue] = []

        for feature_index, signal_id in report.unknown_assignments:
            issues.append(
                _error(
                    PROVENANCE_ISSUE_CODES["unknown_signal"],
                    f"feature #{feature_index} cites signal {signal_id} which is not in the input set",
                    field=f"features.{feature_index}.source_signal_ids",
                )
            )
        for signal_id in report.duplicate_assignments:
            issues.append(
                _error(
                    PROVENANCE_ISSUE_CODES["duplicate"],
                    f"signal {signal_id} is assigned to more than one feature",
                )
            )
        for signal_id in report.unknown_unassigned:
            issues.append(
                _error(
                    PROVENANCE_ISSUE_CODES["unknown_unassigned"],
                    f"unassigned signal {signal_id} is not in the input set",
                    field="unassigned_signal_ids",
                )
            )
        for signal_id in report.assigned_and_unassigned:
            issues.append(
                _error(
                    PROVENANCE_ISSUE_CODES["both"],
                    f"signal {signal_id} appears in both a feature and unassigned_signal_ids",
                )
            )
        for signal_id in report.missing:
            issues.append(
                _error(
                    PROVENANCE_ISSUE_CODES["missing"],
                    f"input signal {signal_id} is neither assigned to a feature nor listed as unassigned",
                )
            )

        return ValidationResult.failure(issues)


def _error(code: str, message: str, *, field: str | None = None) -> ValidationIssue:
    return ValidationIssue(code=code, message=message, field=field, severity=Severity.ERROR, hint=_HINT)
