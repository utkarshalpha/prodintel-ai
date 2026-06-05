"""Tests for the deterministic conflict-integrity gate and its validator adapter."""

from __future__ import annotations

from uuid import uuid4

from app.ai_contracts.base import ModelMeta
from app.ai_contracts.stage3_conflict import (
    ConflictContract,
    ConflictDetectionContract,
    ConflictPosition,
)
from app.ai_contracts.validation.conflict_integrity import validate_conflict_integrity
from app.stages.stage3.validators import CONFLICT_ISSUE_CODES, ConflictIntegrityValidator

F1 = uuid4()
S1, S2, S_UNKNOWN = uuid4(), uuid4(), uuid4()


def _conflict(
    *,
    subject=F1,
    stakeholders=("sales", "engineering"),
    positions=None,
    evidence=(S1, S2),
):
    if positions is None:
        positions = [
            ConflictPosition(stakeholder="sales", stance="advocate", summary="x", evidence_signal_ids=[S1]),
            ConflictPosition(stakeholder="engineering", stance="risk_flag", summary="y", evidence_signal_ids=[S2]),
        ]
    return ConflictContract(
        conflict_id="c1",
        conflict_type="risk",
        severity=4,
        subject_type="feature",
        subject_id=subject,
        stakeholders=list(stakeholders),
        positions=positions,
        evidence_signal_ids=list(evidence),
        confidence={"score": 0.8},
    )


def _detection(conflicts, model_meta: ModelMeta):
    return ConflictDetectionContract(conflicts=conflicts, model_meta=model_meta)


class _Ctx:
    def __init__(self, features, signals):
        self._f, self._s = features, signals

    @property
    def input_feature_ids(self):
        return self._f

    @property
    def input_signal_ids(self):
        return self._s


# --------------------------------------------------------------------------- #
# Deterministic gate
# --------------------------------------------------------------------------- #
def test_sound_conflict_passes(model_meta: ModelMeta) -> None:
    report = validate_conflict_integrity(_detection([_conflict()], model_meta), [F1], [S1, S2])
    assert report.passed is True


def test_empty_conflicts_passes(model_meta: ModelMeta) -> None:
    report = validate_conflict_integrity(_detection([], model_meta), [F1], [S1, S2])
    assert report.passed is True


def test_unknown_subject_detected(model_meta: ModelMeta) -> None:
    report = validate_conflict_integrity(_detection([_conflict(subject=uuid4())], model_meta), [F1], [S1, S2])
    assert report.unknown_subjects and report.passed is False


def test_unknown_evidence_detected(model_meta: ModelMeta) -> None:
    positions = [
        ConflictPosition(stakeholder="sales", stance="advocate", summary="x", evidence_signal_ids=[S_UNKNOWN]),
        ConflictPosition(stakeholder="engineering", stance="oppose", summary="y", evidence_signal_ids=[S2]),
    ]
    report = validate_conflict_integrity(
        _detection([_conflict(positions=positions, evidence=(S_UNKNOWN, S2))], model_meta), [F1], [S1, S2]
    )
    assert any(sid == S_UNKNOWN for _, sid in report.unknown_evidence)


def test_missing_opposition_detected(model_meta: ModelMeta) -> None:
    positions = [
        ConflictPosition(stakeholder="sales", stance="advocate", summary="x", evidence_signal_ids=[S1]),
        ConflictPosition(stakeholder="engineering", stance="advocate", summary="y", evidence_signal_ids=[S2]),
    ]
    report = validate_conflict_integrity(_detection([_conflict(positions=positions)], model_meta), [F1], [S1, S2])
    assert report.missing_opposition and report.passed is False


def test_duplicate_stakeholders_detected(model_meta: ModelMeta) -> None:
    positions = [
        ConflictPosition(stakeholder="sales", stance="advocate", summary="x", evidence_signal_ids=[S1]),
        ConflictPosition(stakeholder="sales", stance="oppose", summary="y", evidence_signal_ids=[S2]),
    ]
    report = validate_conflict_integrity(
        _detection([_conflict(stakeholders=("sales", "sales"), positions=positions)], model_meta), [F1], [S1, S2]
    )
    assert report.duplicate_stakeholders


# --------------------------------------------------------------------------- #
# Validator adapter
# --------------------------------------------------------------------------- #
def test_validator_passes_sound(model_meta: ModelMeta) -> None:
    result = ConflictIntegrityValidator().validate(_detection([_conflict()], model_meta), _Ctx([F1], [S1, S2]))
    assert result.ok is True


def test_validator_reports_unknown_evidence(model_meta: ModelMeta) -> None:
    positions = [
        ConflictPosition(stakeholder="sales", stance="advocate", summary="x", evidence_signal_ids=[S_UNKNOWN]),
        ConflictPosition(stakeholder="engineering", stance="oppose", summary="y", evidence_signal_ids=[S2]),
    ]
    detection = _detection([_conflict(positions=positions, evidence=(S_UNKNOWN, S2))], model_meta)
    result = ConflictIntegrityValidator().validate(detection, _Ctx([F1], [S1, S2]))
    assert result.ok is False
    assert any(i.code == CONFLICT_ISSUE_CODES["unknown_evidence"] for i in result.errors)
