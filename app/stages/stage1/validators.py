"""Stage 1 semantic validator: grounding.

Adapts the deterministic grounding gate
(:func:`app.ai_contracts.validation.grounding.validate_parsed_signal_grounding`)
to the harness :class:`~app.ai_runtime.interfaces.StageValidator` protocol. No
grounding logic is reimplemented here -- this is a thin adapter that turns the
grounding report into harness :class:`ValidationResult` issues.

Policy: **every** extracted claim must be grounded. If any claim fails, the whole
parse is rejected and the harness retries with deterministic, per-claim feedback so
the model can correct the offending spans or drop the unsupported claims. This is
stricter than silently dropping claims and, because contracts are immutable, it is
also the only way to guarantee the *persisted* contract contains nothing
ungrounded.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.ai_contracts.stage1_signal import ParsedSignalContract
from app.ai_contracts.validation.grounding import (
    DEFAULT_CLAIM_THRESHOLD,
    validate_parsed_signal_grounding,
)
from app.ai_runtime.validation_result import Severity, ValidationIssue, ValidationResult

__all__ = ["GroundingValidator", "GROUNDING_ISSUE_CODE"]

#: Stable machine code for an ungrounded-claim issue (used in metrics/assertions).
GROUNDING_ISSUE_CODE = "grounding.ungrounded_claim"

_GROUNDING_HINT = (
    "Set source_span to the exact [start, end) character offsets in the signal text "
    "whose substring contains the words proving this claim, or remove the claim if "
    "the text does not support it."
)


@runtime_checkable
class SupportsSignalText(Protocol):
    """Structural type for any context exposing the analyzed signal text."""

    raw_text: str


class GroundingValidator:
    """Semantic gate that verifies every claim's span against the signal text.

    Parameters
    ----------
    threshold:
        Minimum token overlap for a claim to count as grounded. Defaults to the
        contract spec's :data:`DEFAULT_CLAIM_THRESHOLD`; exposed so the eval harness
        can sweep it.
    """

    def __init__(self, threshold: float = DEFAULT_CLAIM_THRESHOLD) -> None:
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        """The grounding overlap threshold in use."""

        return self._threshold

    def validate(
        self,
        contract: ParsedSignalContract,
        context: SupportsSignalText,
    ) -> ValidationResult:
        """Return a passing result iff every claim is grounded, else per-claim errors.

        Parameters
        ----------
        contract:
            The schema-valid Stage 1 output to verify.
        context:
            Any object exposing ``raw_text`` (the immutable signal text). In
            production this is the stage's ``Stage1Context``.
        """

        report = validate_parsed_signal_grounding(
            contract, context.raw_text, threshold=self._threshold
        )

        # Strict policy: any rejected claim fails the parse.
        if not report.rejected:
            return ValidationResult.success()

        # Map rejected claims back to their positional index for precise feedback.
        rejected_by_identity = {id(claim): result for claim, result in report.rejected}
        issues: list[ValidationIssue] = []
        for index, claim in enumerate(contract.extracted_claims):
            result = rejected_by_identity.get(id(claim))
            if result is None:
                continue
            issues.append(
                ValidationIssue(
                    code=GROUNDING_ISSUE_CODE,
                    message=f"claim #{index} ({claim.text!r}) is not grounded: {result.reason}",
                    field=f"extracted_claims.{index}.source_span",
                    severity=Severity.ERROR,
                    hint=_GROUNDING_HINT,
                    context={
                        "status": result.status.value,
                        "overlap": result.overlap,
                        "source_span": list(claim.source_span),
                        "matched_text": result.matched_text,
                    },
                )
            )
        return ValidationResult.failure(issues)
