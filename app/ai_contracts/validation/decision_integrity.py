"""Stage 4 decision-integrity gate -- the decision-synthesis trust boundary.

Pure, deterministic verification that a :class:`DecisionSynthesisContract` is
evidence-traceable and that each decision honestly accounts for the conflicts over
its subject. It answers: did the model decide about a feature that was not an input,
cite a non-existent signal, acknowledge a conflict that does not exist (or one about
a different feature), silently ignore a known conflict over the feature it is
deciding, or decide the same feature twice?

A contract is sound when, relative to the input feature, signal, and conflict sets,
every decision satisfies:

1. ``subject_id`` is a real input feature (no fabricated subject),
2. no two decisions share the same subject (one decision per feature),
3. every evidence id is a real input signal (every claim traceable; no unsupported
   evidence),
4. every acknowledged conflict id is a real input conflict whose subject is *this*
   decision's subject (no fabricated or mis-attributed conflicts), and
5. every input conflict over this decision's subject is acknowledged -- a decision
   cannot silently ignore a known conflict about the feature it decides.

Like the grounding, provenance, and conflict-integrity gates, this is
dependency-free and returns a structured report; the adapter
(:mod:`app.stages.stage4.validators`) turns it into validation issues.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping
from uuid import UUID

from app.ai_contracts.stage4_decision import DecisionSynthesisContract

__all__ = ["DecisionIntegrityReport", "validate_decision_integrity"]


@dataclass(frozen=True)
class DecisionIntegrityReport:
    """Categorized integrity problems for a decision-synthesis contract.

    Every tuple is empty when the contract is sound.
    """

    unknown_subjects: tuple[tuple[int, UUID], ...]            # (decision_index, subject_id)
    duplicate_subjects: tuple[int, ...]                       # decision indices reusing a subject
    unknown_evidence: tuple[tuple[int, UUID], ...]            # (decision_index, fabricated signal_id)
    unknown_conflicts: tuple[tuple[int, UUID], ...]           # (decision_index, fabricated conflict_id)
    conflict_subject_mismatch: tuple[tuple[int, UUID], ...]   # (decision_index, conflict about another subject)
    unacknowledged_conflicts: tuple[tuple[int, UUID], ...]    # (decision_index, ignored conflict over the subject)

    @property
    def passed(self) -> bool:
        """``True`` iff every category of problem is empty."""

        return not (
            self.unknown_subjects
            or self.duplicate_subjects
            or self.unknown_evidence
            or self.unknown_conflicts
            or self.conflict_subject_mismatch
            or self.unacknowledged_conflicts
        )


def validate_decision_integrity(
    contract: DecisionSynthesisContract,
    input_feature_ids: Iterable[UUID],
    input_signal_ids: Iterable[UUID],
    conflict_subjects: Mapping[UUID, UUID],
) -> DecisionIntegrityReport:
    """Verify subject validity, evidence traceability, and conflict accounting.

    Parameters
    ----------
    contract:
        The schema-valid Stage 4 output to check.
    input_feature_ids:
        Feature ids submitted for synthesis (valid subjects).
    input_signal_ids:
        Signal ids available as evidence (the universe).
    conflict_subjects:
        Mapping of each input conflict id to the feature id it is about. Used both to
        validate acknowledged conflicts and to require that every conflict over a
        decision's subject is acknowledged.
    """

    feature_set = set(input_feature_ids)
    signal_set = set(input_signal_ids)
    conflict_ids = set(conflict_subjects)

    unknown_subjects: list[tuple[int, UUID]] = []
    duplicate_subjects: list[int] = []
    unknown_evidence: list[tuple[int, UUID]] = []
    unknown_conflicts: list[tuple[int, UUID]] = []
    conflict_subject_mismatch: list[tuple[int, UUID]] = []
    unacknowledged_conflicts: list[tuple[int, UUID]] = []

    seen_subjects: set[UUID] = set()

    for index, decision in enumerate(contract.decisions):
        subject = decision.subject_id
        if subject not in feature_set:
            unknown_subjects.append((index, subject))
        if subject in seen_subjects:
            duplicate_subjects.append(index)
        seen_subjects.add(subject)

        for signal_id in decision.evidence_signal_ids:
            if signal_id not in signal_set:
                unknown_evidence.append((index, signal_id))

        acknowledged = set(decision.acknowledged_conflict_ids)
        for conflict_id in decision.acknowledged_conflict_ids:
            if conflict_id not in conflict_ids:
                unknown_conflicts.append((index, conflict_id))
            elif conflict_subjects[conflict_id] != subject:
                conflict_subject_mismatch.append((index, conflict_id))

        # Every conflict over this decision's subject must be acknowledged.
        required = {cid for cid, subj in conflict_subjects.items() if subj == subject}
        for conflict_id in required - acknowledged:
            unacknowledged_conflicts.append((index, conflict_id))

    return DecisionIntegrityReport(
        unknown_subjects=tuple(unknown_subjects),
        duplicate_subjects=tuple(duplicate_subjects),
        unknown_evidence=tuple(dict.fromkeys(unknown_evidence)),
        unknown_conflicts=tuple(dict.fromkeys(unknown_conflicts)),
        conflict_subject_mismatch=tuple(dict.fromkeys(conflict_subject_mismatch)),
        unacknowledged_conflicts=tuple(dict.fromkeys(unacknowledged_conflicts)),
    )
