"""Contract-validation tests for Stage 4 (Decision Synthesis)."""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.ai_contracts.base import ModelMeta
from app.ai_contracts.stage4_decision import DecisionContract, DecisionSynthesisContract

S1, S2 = uuid4(), uuid4()
F = uuid4()
C = uuid4()


def _decision(**overrides):
    data = {
        "decision_id": "d1",
        "subject_type": "feature",
        "subject_id": F,
        "recommendation": "build_now",
        "title": "Build enterprise SSO",
        "rationale": "Sales evidence supports it and the engineering risk is acknowledged.",
        "priority_rank": 1,
        "acknowledged_conflict_ids": [C],
        "evidence_signal_ids": [S1, S2],
        "confidence": {"score": 0.8},
    }
    data.update(overrides)
    return DecisionContract(**data)


def test_valid_decision() -> None:
    decision = _decision()
    assert decision.priority_rank == 1
    assert decision.recommendation.value == "build_now"
    assert decision.confidence.basis.value == "strong"  # derived from score 0.8


def test_decision_requires_evidence() -> None:
    with pytest.raises(ValidationError):
        _decision(evidence_signal_ids=[])


def test_acknowledged_conflicts_may_be_empty() -> None:
    decision = _decision(acknowledged_conflict_ids=[])
    assert decision.acknowledged_conflict_ids == []


def test_priority_rank_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        _decision(priority_rank=0)


def test_rationale_minimum_length_enforced() -> None:
    with pytest.raises(ValidationError):
        _decision(rationale="short")


def test_recommendation_is_closed_set() -> None:
    with pytest.raises(ValidationError):
        _decision(recommendation="maybe")


def test_synthesis_requires_at_least_one_decision(model_meta: ModelMeta) -> None:
    with pytest.raises(ValidationError):
        DecisionSynthesisContract(model_meta=model_meta, decisions=[])


def test_synthesis_accepts_decisions(model_meta: ModelMeta) -> None:
    contract = DecisionSynthesisContract(model_meta=model_meta, decisions=[_decision()])
    assert len(contract.decisions) == 1
    assert contract.schema_version == "1.0"


def test_synthesis_rejects_unknown_field(model_meta: ModelMeta) -> None:
    with pytest.raises(ValidationError):
        DecisionSynthesisContract(model_meta=model_meta, decisions=[_decision()], surprise=1)
