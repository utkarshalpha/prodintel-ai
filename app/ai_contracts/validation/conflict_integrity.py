"""Stage 3 conflict-integrity gate -- the conflict-detection trust boundary.

Pure, deterministic verification that a :class:`ConflictDetectionContract` is
evidence-traceable and that each claimed conflict is a real opposition. It answers:
did the model invent a subject, cite a non-existent signal, list stakeholders that
do not match the positions, or assert a "conflict" where everyone actually agrees?

A contract is sound when, relative to the input feature and signal sets, every
conflict satisfies:

1. ``subject_id`` is a real input feature (no fabricated subject),
2. its ``stakeholders`` are distinct and equal the set of position stakeholders,
3. every position's evidence (and the aggregate evidence) are real input signals
   (every claim traceable; no unsupported evidence), and
4. the positions actually oppose -- at least two distinct non-neutral stances
   (advocate vs oppose/risk_flag, etc.); otherwise it is not a conflict.

An empty conflict list is sound (stakeholders agree). Like the grounding and
provenance gates, this is dependency-free and returns a structured report; the
adapter (:mod:`app.stages.stage3.validators`) turns it into validation issues.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from uuid import UUID

from app.ai_contracts.enums import Stance
from app.ai_contracts.stage3_conflict import ConflictDetectionContract

__all__ = ["ConflictIntegrityReport", "validate_conflict_integrity"]

# Stances that count as taking a side; neutral does not create opposition.
_NON_NEUTRAL = frozenset({Stance.ADVOCATE, Stance.OPPOSE, Stance.RISK_FLAG})


@dataclass(frozen=True)
class ConflictIntegrityReport:
    """Categorized integrity problems for a conflict-detection contract.

    Every tuple is empty when the contract is sound (including the empty-conflict
    case).
    """

    unknown_subjects: tuple[tuple[int, UUID], ...]          # (conflict_index, subject_id)
    duplicate_stakeholders: tuple[int, ...]                 # conflict indices with repeated stakeholders
    stakeholder_mismatch: tuple[int, ...]                   # positions set != declared stakeholders
    unknown_evidence: tuple[tuple[int, UUID], ...]          # (conflict_index, fabricated signal_id)
    missing_opposition: tuple[int, ...]                     # conflict indices lacking opposing stances

    @property
    def passed(self) -> bool:
        """``True`` iff every category of problem is empty."""

        return not (
            self.unknown_subjects
            or self.duplicate_stakeholders
            or self.stakeholder_mismatch
            or self.unknown_evidence
            or self.missing_opposition
        )


def validate_conflict_integrity(
    contract: ConflictDetectionContract,
    input_feature_ids: Iterable[UUID],
    input_signal_ids: Iterable[UUID],
) -> ConflictIntegrityReport:
    """Verify subject validity, evidence traceability, and genuine opposition.

    Parameters
    ----------
    contract:
        The schema-valid Stage 3 output to check.
    input_feature_ids:
        Feature ids that were submitted for conflict detection (valid subjects).
    input_signal_ids:
        Signal ids available as evidence (the universe).
    """

    feature_set = set(input_feature_ids)
    signal_set = set(input_signal_ids)

    unknown_subjects: list[tuple[int, UUID]] = []
    duplicate_stakeholders: list[int] = []
    stakeholder_mismatch: list[int] = []
    unknown_evidence: list[tuple[int, UUID]] = []
    missing_opposition: list[int] = []

    for index, conflict in enumerate(contract.conflicts):
        if conflict.subject_id not in feature_set:
            unknown_subjects.append((index, conflict.subject_id))

        declared = list(conflict.stakeholders)
        if len(set(declared)) != len(declared):
            duplicate_stakeholders.append(index)

        position_stakeholders = {position.stakeholder for position in conflict.positions}
        if position_stakeholders != set(declared):
            stakeholder_mismatch.append(index)

        for position in conflict.positions:
            for signal_id in position.evidence_signal_ids:
                if signal_id not in signal_set:
                    unknown_evidence.append((index, signal_id))
        for signal_id in conflict.evidence_signal_ids:
            if signal_id not in signal_set:
                unknown_evidence.append((index, signal_id))

        sides = {position.stance for position in conflict.positions} & _NON_NEUTRAL
        if len(sides) < 2:
            missing_opposition.append(index)

    return ConflictIntegrityReport(
        unknown_subjects=tuple(unknown_subjects),
        duplicate_stakeholders=tuple(duplicate_stakeholders),
        stakeholder_mismatch=tuple(stakeholder_mismatch),
        unknown_evidence=tuple(dict.fromkeys(unknown_evidence)),  # dedupe, preserve order
        missing_opposition=tuple(missing_opposition),
    )
