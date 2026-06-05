"""Stage 2 -- Feature Extraction contracts.

The model clusters many grounded ``ParsedSignal``s into a smaller set of normalized
product features. Because one model call produces *many* features, the harness root
is :class:`FeatureExtractionContract` (a list of features), and an individual feature
is the value object :class:`FeatureContract`.

Provenance is the keystone, exactly as grounding was for Stage 1: every feature must
cite the signal ids it derives from, and the model must account for every input
signal (assigned to one feature or explicitly unassigned). The *structural* rules
(>= 1 feature, >= 1 source per feature) live here; the *semantic* rule -- that those
ids are real input ids and partition the input -- lives in the provenance validator,
which has the input set.

Harness-managed metadata: ``model_meta``/``schema_version`` are carried on the
extraction root and injected by the runner from the real response (the Stage 1
pattern), so the model never fabricates per-feature provenance metadata.
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from pydantic import Field

from app.ai_contracts.base import AIContractBase, ConfidenceBlock, FrozenModel

__all__ = ["FeatureContract", "FeatureExtractionContract"]


class FeatureContract(FrozenModel):
    """One normalized product feature derived from clustered signals.

    ``feature_id`` is a model-local correlation handle for this run (e.g. ``"f1"``),
    not the persisted database id -- the database assigns the durable UUID on insert.
    ``source_signal_ids`` is the provenance: the signals this feature was clustered
    from, verified against the input set by the provenance validator.
    """

    feature_id: str = Field(..., min_length=1, description="Model-local handle for this feature (not the DB id).")
    title: str = Field(..., min_length=1, max_length=120, description="Normalized, deduplicated feature name.")
    description: str = Field(..., min_length=1, description="What the feature is and why it is needed.")
    jtbd: str = Field(
        ...,
        min_length=10,
        description="Jobs-to-be-Done statement: 'When <situation>, I want <motivation>, so I can <outcome>'.",
    )
    source_signal_ids: list[UUID] = Field(
        ...,
        min_length=1,
        description="Signals this feature derives from (provenance). Must be a subset of the input.",
    )
    confidence: ConfidenceBlock = Field(..., description="Decomposed confidence for this feature.")


class FeatureExtractionContract(AIContractBase):
    """The Stage 2 tool output: the full set of features plus unassigned signals.

    Forcing ``unassigned_signal_ids`` to be explicit makes the model *account for*
    every input signal rather than silently dropping inconvenient ones.
    """

    features: list[FeatureContract] = Field(..., min_length=1, description="At least one extracted feature.")
    unassigned_signal_ids: list[UUID] = Field(
        default_factory=list,
        description="Input signals that fit no feature -- explicit, never silently dropped.",
    )
