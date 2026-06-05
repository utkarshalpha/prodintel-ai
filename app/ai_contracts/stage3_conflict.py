"""Stage 3 -- Conflict Detection contracts.

The model inspects a set of features and the stakeholder signals behind them, and
emits the *disagreements* between stakeholders. One model call can surface many
conflicts -- or **none** (stakeholders may simply agree) -- so the harness root
:class:`ConflictDetectionContract` carries a possibly-empty list, and an individual
conflict is the value object :class:`ConflictContract`.

Evidence traceability is the keystone, exactly as grounding (Stage 1) and provenance
(Stage 2): every stakeholder *position* must cite the signals that substantiate it,
and the conflict must reference a real subject feature. The structural rules
(>= 2 stakeholders, >= 2 positions, >= 1 evidence id per position, bounded severity
and confidence) live here; the semantic rules -- that the cited ids are real inputs
and that the positions actually oppose -- live in the conflict-integrity validator,
which has the input sets.

Harness-managed metadata: ``model_meta``/``schema_version`` are carried on the root
and injected by the runner from the real response (the Stage 1/2 pattern).
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from pydantic import Field

from app.ai_contracts.base import AIContractBase, ConfidenceBlock, FrozenModel
from app.ai_contracts.enums import ConflictType, Stance, StakeholderType, SubjectType

__all__ = ["ConflictPosition", "ConflictContract", "ConflictDetectionContract"]


class ConflictPosition(FrozenModel):
    """One stakeholder's stance in a conflict, with its supporting evidence.

    ``evidence_signal_ids`` is what makes the position traceable: the signals whose
    content backs this stance. The integrity validator verifies they are real input
    signals; a position with no evidence is rejected at construction.
    """

    stakeholder: StakeholderType = Field(..., description="The stakeholder holding this position.")
    stance: Stance = Field(..., description="advocate | oppose | risk_flag | neutral.")
    summary: str = Field(..., min_length=1, description="One-line statement of this stakeholder's position.")
    evidence_signal_ids: list[UUID] = Field(
        ...,
        min_length=1,
        description="Signals substantiating this position (subset of the input signals).",
    )


class ConflictContract(FrozenModel):
    """A single detected conflict between stakeholders over a subject feature.

    ``conflict_id`` is a model-local correlation handle for this run, not the
    persisted database id. The conflict must involve at least two stakeholders with
    genuinely opposing positions; the integrity validator enforces opposition.
    """

    conflict_id: str = Field(..., min_length=1, description="Model-local handle for this conflict (not the DB id).")
    conflict_type: ConflictType = Field(..., description="priority | risk | resource | strategic.")
    severity: Annotated[int, Field(ge=1, le=5)] = Field(..., description="Severity from 1 (minor) to 5 (severe).")
    subject_type: SubjectType = Field(..., description="What the conflict is about (feature | objective).")
    subject_id: UUID = Field(..., description="Id of the subject (a feature id) -- must be an input feature.")
    stakeholders: list[StakeholderType] = Field(
        ...,
        min_length=2,
        description="The distinct stakeholders in conflict (matches the positions).",
    )
    positions: list[ConflictPosition] = Field(
        ...,
        min_length=2,
        description="Each stakeholder's stance and evidence; must contain an opposition.",
    )
    evidence_signal_ids: list[UUID] = Field(
        ...,
        min_length=1,
        description="Aggregate of all evidence signals across positions (subset of input).",
    )
    confidence: ConfidenceBlock = Field(..., description="Decomposed confidence for this conflict.")


class ConflictDetectionContract(AIContractBase):
    """The Stage 3 tool output: the set of detected conflicts (possibly empty).

    An empty list is a valid, meaningful result: it asserts the stakeholders do not
    conflict over the analyzed features. The validator therefore passes an empty
    list, and the runner treats it as success.
    """

    conflicts: list[ConflictContract] = Field(
        default_factory=list,
        description="Detected conflicts; empty when stakeholders do not disagree.",
    )
