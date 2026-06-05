"""Contract-validation tests for Stage 3 (Conflict Detection)."""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.ai_contracts.base import ModelMeta
from app.ai_contracts.stage3_conflict import (
    ConflictContract,
    ConflictDetectionContract,
    ConflictPosition,
)

A, B = uuid4(), uuid4()
F = uuid4()


def _position(stakeholder="sales", stance="advocate", evidence=None):
    return ConflictPosition(
        stakeholder=stakeholder,
        stance=stance,
        summary="a position",
        evidence_signal_ids=evidence if evidence is not None else [A],
    )


def _conflict(**overrides):
    data = {
        "conflict_id": "c1",
        "conflict_type": "risk",
        "severity": 4,
        "subject_type": "feature",
        "subject_id": F,
        "stakeholders": ["sales", "engineering"],
        "positions": [_position("sales", "advocate", [A]), _position("engineering", "risk_flag", [B])],
        "evidence_signal_ids": [A, B],
        "confidence": {"score": 0.8},
    }
    data.update(overrides)
    return ConflictContract(**data)


def test_valid_conflict() -> None:
    conflict = _conflict()
    assert conflict.severity == 4
    assert len(conflict.positions) == 2


def test_position_requires_evidence() -> None:
    with pytest.raises(ValidationError):
        _position(evidence=[])


def test_conflict_requires_two_stakeholders() -> None:
    with pytest.raises(ValidationError):
        _conflict(stakeholders=["sales"])


def test_conflict_requires_two_positions() -> None:
    with pytest.raises(ValidationError):
        _conflict(positions=[_position()])


def test_severity_must_be_bounded() -> None:
    with pytest.raises(ValidationError):
        _conflict(severity=6)
    with pytest.raises(ValidationError):
        _conflict(severity=0)


def test_detection_allows_empty_conflicts(model_meta: ModelMeta) -> None:
    contract = ConflictDetectionContract(model_meta=model_meta)
    assert contract.conflicts == []
    assert contract.schema_version == "1.0"


def test_detection_rejects_unknown_field(model_meta: ModelMeta) -> None:
    with pytest.raises(ValidationError):
        ConflictDetectionContract(model_meta=model_meta, conflicts=[], surprise=1)
