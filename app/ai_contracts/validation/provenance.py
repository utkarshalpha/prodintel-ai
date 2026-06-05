"""Stage 2 provenance gate -- the feature-extraction trust boundary.

Pure, deterministic verification that a :class:`FeatureExtractionContract` correctly
partitions and attributes the input signals. It answers: did the model invent
provenance, double-count a signal, or drop one?

A contract is sound when, relative to the input signal set:

1. every cited ``source_signal_id`` is a real input id (no fabricated provenance),
2. no signal is assigned to more than one feature,
3. every ``unassigned_signal_id`` is a real input id,
4. no signal is both assigned and unassigned, and
5. every input signal is accounted for (assigned or unassigned) -- none missing.

Like the grounding gate, this is dependency-free and returns a structured report;
the harness adapter (see :mod:`app.stages.stage2.validators`) turns it into
validation issues and deterministic retry feedback.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable
from uuid import UUID

from app.ai_contracts.stage2_feature import FeatureExtractionContract

__all__ = ["FeatureProvenanceReport", "validate_feature_provenance"]


@dataclass(frozen=True)
class FeatureProvenanceReport:
    """Structured result of checking feature provenance against the input set.

    Every tuple is empty when the contract is sound.
    """

    unknown_assignments: tuple[tuple[int, UUID], ...]   # (feature_index, fabricated signal id)
    duplicate_assignments: tuple[UUID, ...]             # signals assigned to >1 feature
    unknown_unassigned: tuple[UUID, ...]                # unassigned ids not in the input
    assigned_and_unassigned: tuple[UUID, ...]           # ids in both assigned and unassigned
    missing: tuple[UUID, ...]                            # input ids in neither

    @property
    def passed(self) -> bool:
        """``True`` iff every category of problem is empty."""

        return not (
            self.unknown_assignments
            or self.duplicate_assignments
            or self.unknown_unassigned
            or self.assigned_and_unassigned
            or self.missing
        )


def validate_feature_provenance(
    contract: FeatureExtractionContract,
    input_signal_ids: Iterable[UUID],
) -> FeatureProvenanceReport:
    """Verify that ``contract`` faithfully partitions and attributes the inputs.

    Parameters
    ----------
    contract:
        The schema-valid Stage 2 output to check.
    input_signal_ids:
        The signal ids that were fed into extraction (the universe).

    Returns
    -------
    FeatureProvenanceReport
        The categorized problems (all empty when sound).
    """

    input_set = set(input_signal_ids)

    unknown_assignments: list[tuple[int, UUID]] = []
    assignment_counts: Counter[UUID] = Counter()
    for index, feature in enumerate(contract.features):
        for signal_id in feature.source_signal_ids:
            assignment_counts[signal_id] += 1
            if signal_id not in input_set:
                unknown_assignments.append((index, signal_id))

    assigned_set = set(assignment_counts)
    duplicate_assignments = sorted(
        (sid for sid, count in assignment_counts.items() if count > 1),
        key=str,
    )

    unassigned_set = set(contract.unassigned_signal_ids)
    unknown_unassigned = sorted(unassigned_set - input_set, key=str)
    assigned_and_unassigned = sorted(assigned_set & unassigned_set, key=str)
    missing = sorted(input_set - assigned_set - unassigned_set, key=str)

    return FeatureProvenanceReport(
        unknown_assignments=tuple(unknown_assignments),
        duplicate_assignments=tuple(duplicate_assignments),
        unknown_unassigned=tuple(unknown_unassigned),
        assigned_and_unassigned=tuple(assigned_and_unassigned),
        missing=tuple(missing),
    )
