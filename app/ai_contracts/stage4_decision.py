"""Stage 4 -- Decision Synthesis contracts.

The model takes the product features, the stakeholder signals behind them, and the
conflicts already detected over them, and emits *evidence-backed product decisions*:
for each subject feature, what to do (a recommendation), why (a rationale), how it
ranks, which conflicts it accounts for, and which signals substantiate it. One model
call produces *many* decisions, so the harness root is
:class:`DecisionSynthesisContract` (a non-empty list) and an individual decision is
the value object :class:`DecisionContract`.

Evidence traceability is the keystone, exactly as grounding (Stage 1), provenance
(Stage 2), and conflict integrity (Stage 3): every decision must cite the signals
that substantiate it and the conflicts it acknowledges, and its subject must be a
real input feature. The *structural* rules (>= 1 decision, >= 1 evidence id per
decision, bounded rank and confidence) live here; the *semantic* rules -- that the
cited ids are real inputs, that a decision acknowledges every conflict over its
subject, and that subjects are not decided twice -- live in the decision-integrity
validator, which has the input sets.

Harness-managed metadata: ``model_meta``/``schema_version`` are carried on the root
and injected by the runner from the real response (the Stage 1/2/3 pattern), so the
model never fabricates per-decision provenance metadata.
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from pydantic import Field

from app.ai_contracts.base import AIContractBase, ConfidenceBlock, FrozenModel
from app.ai_contracts.enums import DecisionRecommendation, SubjectType

__all__ = ["DecisionContract", "DecisionSynthesisContract"]


class DecisionContract(FrozenModel):
    """One synthesized, evidence-backed decision about a subject feature.

    ``decision_id`` is a model-local correlation handle for this run, not the
    persisted database id. ``evidence_signal_ids`` is what makes the decision
    traceable; ``acknowledged_conflict_ids`` are the detected conflicts this decision
    accounts for. The integrity validator verifies the subject is a real feature, the
    evidence/conflicts are real inputs, and that no conflict over the subject is left
    unacknowledged.
    """

    decision_id: str = Field(..., min_length=1, description="Model-local handle for this decision (not the DB id).")
    subject_type: SubjectType = Field(..., description="What the decision is about (feature | objective).")
    subject_id: UUID = Field(..., description="Id of the subject (a feature id) -- must be an input feature.")
    recommendation: DecisionRecommendation = Field(
        ...,
        description="The recommended action: build_now | build_later | reject | needs_discussion.",
    )
    title: str = Field(..., min_length=1, max_length=160, description="One-line headline of the decision.")
    rationale: str = Field(
        ...,
        min_length=10,
        description="The reasoning for the recommendation, referencing the evidence and any conflicts.",
    )
    priority_rank: Annotated[int, Field(ge=1)] = Field(
        ...,
        description="Relative priority of this decision (1 = highest). Lower ranks come first.",
    )
    acknowledged_conflict_ids: list[UUID] = Field(
        default_factory=list,
        description="Detected conflicts over this subject that the decision accounts for (subset of inputs).",
    )
    evidence_signal_ids: list[UUID] = Field(
        ...,
        min_length=1,
        description="Signals substantiating this decision (a subset of the input signals).",
    )
    confidence: ConfidenceBlock = Field(..., description="Decomposed confidence for this decision.")


class DecisionSynthesisContract(AIContractBase):
    """The Stage 4 tool output: the set of synthesized decisions.

    At least one decision is required -- the stage is asked to decide over the
    provided features, so an empty result is not a meaningful outcome (unlike
    Stage 3, where "no conflict" is a real answer).
    """

    decisions: list[DecisionContract] = Field(
        ...,
        min_length=1,
        description="At least one synthesized, evidence-backed decision.",
    )
