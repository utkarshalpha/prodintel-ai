"""Tests for the deterministic decision-integrity gate and its validator adapter."""

from __future__ import annotations

from dataclasses import asdict
from uuid import uuid4

from app.ai_contracts.base import ModelMeta
from app.ai_contracts.stage4_decision import DecisionContract, DecisionSynthesisContract
from app.ai_contracts.validation.decision_integrity import validate_decision_integrity
from app.stages.stage4.validators import DECISION_ISSUE_CODES, DecisionIntegrityValidator

F1, F2 = uuid4(), uuid4()
S1, S2, S_UNKNOWN = uuid4(), uuid4(), uuid4()
C1, C_UNKNOWN = uuid4(), uuid4()
K1, K2, K_UNKNOWN = uuid4(), uuid4(), uuid4()  # framework-knowledge chunk ids

# Conflict C1 is about feature F1.
CONFLICT_SUBJECTS = {C1: F1}
FRAMEWORK_POOL = (K1, K2)


def _decision(
    *,
    decision_id="d1",
    subject=F1,
    evidence=(S1, S2),
    acknowledged=(C1,),
    framework=(),
    rank=1,
):
    return DecisionContract(
        decision_id=decision_id,
        subject_type="feature",
        subject_id=subject,
        recommendation="build_now",
        title="Build it",
        rationale="The evidence supports building it and the conflict is acknowledged.",
        priority_rank=rank,
        acknowledged_conflict_ids=list(acknowledged),
        evidence_signal_ids=list(evidence),
        framework_citation_ids=list(framework),
        confidence={"score": 0.8},
    )


def _synthesis(decisions, model_meta: ModelMeta):
    return DecisionSynthesisContract(decisions=decisions, model_meta=model_meta)


class _Ctx:
    def __init__(self, features, signals, conflict_subjects):
        self._f, self._s, self._c = features, signals, conflict_subjects

    @property
    def input_feature_ids(self):
        return self._f

    @property
    def input_signal_ids(self):
        return self._s

    @property
    def conflict_subjects(self):
        return self._c


# --------------------------------------------------------------------------- #
# Deterministic gate
# --------------------------------------------------------------------------- #
def test_sound_decision_passes(model_meta: ModelMeta) -> None:
    report = validate_decision_integrity(
        _synthesis([_decision()], model_meta), [F1], [S1, S2], CONFLICT_SUBJECTS
    )
    assert report.passed is True


def test_no_conflicts_decision_passes(model_meta: ModelMeta) -> None:
    """A feature with no conflicts needs no acknowledgements."""

    report = validate_decision_integrity(
        _synthesis([_decision(subject=F2, acknowledged=())], model_meta), [F2], [S1, S2], {}
    )
    assert report.passed is True


def test_unknown_subject_detected(model_meta: ModelMeta) -> None:
    report = validate_decision_integrity(
        _synthesis([_decision(subject=uuid4(), acknowledged=())], model_meta), [F1], [S1, S2], {}
    )
    assert report.unknown_subjects and report.passed is False


def test_duplicate_subject_detected(model_meta: ModelMeta) -> None:
    decisions = [
        _decision(decision_id="d1", rank=1),
        _decision(decision_id="d2", rank=2),
    ]
    report = validate_decision_integrity(_synthesis(decisions, model_meta), [F1], [S1, S2], CONFLICT_SUBJECTS)
    assert report.duplicate_subjects and report.passed is False


def test_unknown_evidence_detected(model_meta: ModelMeta) -> None:
    report = validate_decision_integrity(
        _synthesis([_decision(evidence=(S_UNKNOWN,))], model_meta), [F1], [S1, S2], CONFLICT_SUBJECTS
    )
    assert any(sid == S_UNKNOWN for _, sid in report.unknown_evidence)
    assert report.passed is False


def test_unknown_conflict_detected(model_meta: ModelMeta) -> None:
    report = validate_decision_integrity(
        _synthesis([_decision(acknowledged=(C1, C_UNKNOWN))], model_meta), [F1], [S1, S2], CONFLICT_SUBJECTS
    )
    assert any(cid == C_UNKNOWN for _, cid in report.unknown_conflicts)
    assert report.passed is False


def test_conflict_subject_mismatch_detected(model_meta: ModelMeta) -> None:
    """Acknowledging a conflict that is about a different feature is a mismatch."""

    report = validate_decision_integrity(
        _synthesis([_decision(subject=F2, acknowledged=(C1,))], model_meta),
        [F1, F2],
        [S1, S2],
        CONFLICT_SUBJECTS,  # C1 is about F1, not F2
    )
    assert report.conflict_subject_mismatch and report.passed is False


def test_unacknowledged_conflict_detected(model_meta: ModelMeta) -> None:
    """A decision over F1 that ignores the conflict C1 over F1 fails."""

    report = validate_decision_integrity(
        _synthesis([_decision(acknowledged=())], model_meta), [F1], [S1, S2], CONFLICT_SUBJECTS
    )
    assert any(cid == C1 for _, cid in report.unacknowledged_conflicts)
    assert report.passed is False


# --------------------------------------------------------------------------- #
# Framework-citation grounding (Phase 7A)
# --------------------------------------------------------------------------- #
def test_empty_framework_citations_valid(model_meta: ModelMeta) -> None:
    """A decision citing no framework passes even when a pool is injected."""

    report = validate_decision_integrity(
        _synthesis([_decision(framework=())], model_meta),
        [F1],
        [S1, S2],
        CONFLICT_SUBJECTS,
        FRAMEWORK_POOL,
    )
    assert report.unknown_framework_citations == () and report.passed is True


def test_grounded_framework_citations_valid(model_meta: ModelMeta) -> None:
    report = validate_decision_integrity(
        _synthesis([_decision(framework=(K1, K2))], model_meta),
        [F1],
        [S1, S2],
        CONFLICT_SUBJECTS,
        FRAMEWORK_POOL,
    )
    assert report.unknown_framework_citations == () and report.passed is True


def test_unknown_framework_citation_fails(model_meta: ModelMeta) -> None:
    report = validate_decision_integrity(
        _synthesis([_decision(framework=(K1, K_UNKNOWN))], model_meta),
        [F1],
        [S1, S2],
        CONFLICT_SUBJECTS,
        FRAMEWORK_POOL,
    )
    assert any(cid == K_UNKNOWN for _, cid in report.unknown_framework_citations)
    assert all(cid != K1 for _, cid in report.unknown_framework_citations)  # grounded one OK
    assert report.passed is False


def test_framework_grounding_does_not_affect_other_rules(model_meta: ModelMeta) -> None:
    """A sound, grounded decision passes every rule; an unrelated failure is unchanged."""

    sound = validate_decision_integrity(
        _synthesis([_decision(framework=(K1,))], model_meta), [F1], [S1, S2], CONFLICT_SUBJECTS, FRAMEWORK_POOL
    )
    assert sound.passed is True

    # An evidence failure is still reported, and is independent of framework grounding.
    bad_evidence = validate_decision_integrity(
        _synthesis([_decision(evidence=(S_UNKNOWN,), framework=(K1,))], model_meta),
        [F1],
        [S1, S2],
        CONFLICT_SUBJECTS,
        FRAMEWORK_POOL,
    )
    assert bad_evidence.unknown_evidence and bad_evidence.unknown_framework_citations == ()
    assert bad_evidence.passed is False


def test_report_serialization_includes_framework_field(model_meta: ModelMeta) -> None:
    grounded = validate_decision_integrity(
        _synthesis([_decision(framework=(K1,))], model_meta), [F1], [S1, S2], CONFLICT_SUBJECTS, FRAMEWORK_POOL
    )
    grounded_dict = asdict(grounded)
    assert grounded_dict["unknown_framework_citations"] == ()

    bad = validate_decision_integrity(
        _synthesis([_decision(framework=(K_UNKNOWN,))], model_meta), [F1], [S1, S2], CONFLICT_SUBJECTS, FRAMEWORK_POOL
    )
    assert asdict(bad)["unknown_framework_citations"] == ((0, K_UNKNOWN),)


def test_backward_compatibility_without_framework_pool(model_meta: ModelMeta) -> None:
    """The 4-arg call (no pool) and the default contract field are unchanged."""

    # Contract default: no framework citations supplied -> empty list.
    assert _decision().framework_citation_ids == []

    # Legacy 4-positional-arg call still validates a sound, framework-less decision.
    report = validate_decision_integrity(
        _synthesis([_decision()], model_meta), [F1], [S1, S2], CONFLICT_SUBJECTS
    )
    assert report.unknown_framework_citations == () and report.passed is True


# --------------------------------------------------------------------------- #
# Validator adapter
# --------------------------------------------------------------------------- #
def test_validator_passes_sound(model_meta: ModelMeta) -> None:
    result = DecisionIntegrityValidator().validate(
        _synthesis([_decision()], model_meta), _Ctx([F1], [S1, S2], CONFLICT_SUBJECTS)
    )
    assert result.ok is True


def test_validator_reports_unacknowledged_conflict(model_meta: ModelMeta) -> None:
    synthesis = _synthesis([_decision(acknowledged=())], model_meta)
    result = DecisionIntegrityValidator().validate(synthesis, _Ctx([F1], [S1, S2], CONFLICT_SUBJECTS))
    assert result.ok is False
    assert any(i.code == DECISION_ISSUE_CODES["unacknowledged_conflict"] for i in result.errors)
