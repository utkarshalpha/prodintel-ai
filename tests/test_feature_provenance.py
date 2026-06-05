"""Tests for the deterministic provenance gate and its validator adapter."""

from __future__ import annotations

from uuid import uuid4

from app.ai_contracts.base import ModelMeta
from app.ai_contracts.stage2_feature import FeatureContract, FeatureExtractionContract
from app.ai_contracts.validation.provenance import validate_feature_provenance
from app.stages.stage2.validators import PROVENANCE_ISSUE_CODES, FeatureProvenanceValidator

A, B, C = uuid4(), uuid4(), uuid4()


def _feature(source_ids, fid="f1"):
    return FeatureContract(
        feature_id=fid,
        title="Feature",
        description="desc",
        jtbd="When x, I want y, so I can z",
        source_signal_ids=source_ids,
        confidence={"score": 0.8},
    )


def _extraction(features, unassigned, model_meta: ModelMeta):
    return FeatureExtractionContract(features=features, unassigned_signal_ids=unassigned, model_meta=model_meta)


# --------------------------------------------------------------------------- #
# Deterministic gate
# --------------------------------------------------------------------------- #
def test_sound_extraction_passes(model_meta: ModelMeta) -> None:
    report = validate_feature_provenance(_extraction([_feature([A, B])], [], model_meta), [A, B])
    assert report.passed is True


def test_unassigned_accounts_for_remaining(model_meta: ModelMeta) -> None:
    report = validate_feature_provenance(_extraction([_feature([A])], [B], model_meta), [A, B])
    assert report.passed is True


def test_unknown_assignment_detected(model_meta: ModelMeta) -> None:
    report = validate_feature_provenance(_extraction([_feature([A, C])], [], model_meta), [A])
    assert any(sid == C for _, sid in report.unknown_assignments)
    assert report.passed is False


def test_duplicate_assignment_detected(model_meta: ModelMeta) -> None:
    extraction = _extraction([_feature([A], "f1"), _feature([A], "f2")], [], model_meta)
    report = validate_feature_provenance(extraction, [A])
    assert A in report.duplicate_assignments


def test_missing_signal_detected(model_meta: ModelMeta) -> None:
    report = validate_feature_provenance(_extraction([_feature([A])], [], model_meta), [A, B])
    assert B in report.missing


def test_signal_in_both_assigned_and_unassigned(model_meta: ModelMeta) -> None:
    report = validate_feature_provenance(_extraction([_feature([A])], [A], model_meta), [A])
    assert A in report.assigned_and_unassigned


def test_unknown_unassigned_detected(model_meta: ModelMeta) -> None:
    report = validate_feature_provenance(_extraction([_feature([A])], [C], model_meta), [A])
    assert C in report.unknown_unassigned


# --------------------------------------------------------------------------- #
# Validator adapter -> ValidationResult
# --------------------------------------------------------------------------- #
class _Ctx:
    def __init__(self, ids):
        self._ids = ids

    @property
    def input_signal_ids(self):
        return self._ids


def test_validator_passes_sound_extraction(model_meta: ModelMeta) -> None:
    result = FeatureProvenanceValidator().validate(_extraction([_feature([A, B])], [], model_meta), _Ctx([A, B]))
    assert result.ok is True


def test_validator_reports_unknown_signal_with_field(model_meta: ModelMeta) -> None:
    result = FeatureProvenanceValidator().validate(_extraction([_feature([A, C])], [], model_meta), _Ctx([A]))
    assert result.ok is False
    issue = result.errors[0]
    assert issue.code == PROVENANCE_ISSUE_CODES["unknown_signal"]
    assert issue.field == "features.0.source_signal_ids"
    assert issue.hint is not None
