"""Stage 4 conflict-completeness tests (HIGH #2).

ADR-012: *"every conflict affecting a subject must be acknowledged."* The review
found that ``DecisionService`` built the required-conflict set via the paginated
``ConflictRepository.list(...)`` (default ``limit=100``), so a subject with >100
conflicts could have some silently excluded -- letting a decision pass validation
while omitting real conflicts.

These tests prove the fix: the unbounded ``list_by_subjects`` path considers *all*
conflicts, the completeness invariant still fails on any omission at scale, and
normal conflict counts are unaffected. All deterministic, no network (fakes only).
"""

from __future__ import annotations

import uuid

import pytest

from app.ai_contracts.enums import ConflictType, StakeholderType, SubjectType
from app.ai_contracts.stage4_decision import DecisionContract, DecisionSynthesisContract
from app.ai_contracts.validation.decision_integrity import validate_decision_integrity
from app.models.conflict import Conflict
from app.repositories.conflict_repository import ConflictRepository
from app.services.errors import DecisionSynthesisFailedError

from tests.app_helpers import make_engine, make_session_factory
from tests.test_decision_service import _decision_service, _feature_service, _signal_service


@pytest.fixture
def session():
    db = make_session_factory(make_engine())()
    try:
        yield db
    finally:
        db.close()


def _bare_conflict(subject_id: uuid.UUID, i: int) -> Conflict:
    """A minimal persistable conflict over ``subject_id`` (no parties needed here)."""

    return Conflict(
        subject_type=SubjectType.FEATURE,
        subject_id=subject_id,
        conflict_type=ConflictType.RISK,
        severity=(i % 5) + 1,
        confidence_score=0.8,
        confidence={"score": 0.8, "basis": "strong", "components": {}},
        model_meta={"source": "test"},
    )


def _feature_with_n_conflicts(session, n: int):
    """Build an analyzed feature (Stage 1→2) and persist ``n`` conflicts over it."""

    signals = _signal_service(session)
    sales = signals.create_signal(source_type=StakeholderType.SALES, raw_text="Enterprise SSO is critical").signal
    eng = signals.create_signal(source_type=StakeholderType.ENGINEERING, raw_text="SSO is high risk to build").signal
    signals.analyze_signal(sales.id)
    signals.analyze_signal(eng.id)
    feature = _feature_service(session).extract_features([sales.id, eng.id]).features[0]
    for i in range(n):
        session.add(_bare_conflict(feature.id, i))
    session.commit()
    return feature


# --------------------------------------------------------------------------- #
# Root cause: the repository path
# --------------------------------------------------------------------------- #
def test_list_by_subjects_returns_all_conflicts_with_no_limit(session) -> None:
    """The unbounded accessor returns all 101; the paginated listing still caps at 100."""

    subject = uuid.uuid4()
    for i in range(101):
        session.add(_bare_conflict(subject, i))
    session.commit()

    repo = ConflictRepository(session)
    assert len(repo.list_by_subjects([subject])) == 101          # fix: no truncation
    assert len(repo.list(subject_id=subject)) == 100             # root cause: default limit caps


def test_list_by_subjects_handles_empty_input(session) -> None:
    assert ConflictRepository(session).list_by_subjects([]) == []


# --------------------------------------------------------------------------- #
# A) 101+ conflicts are all considered (end-to-end through synthesis)
# --------------------------------------------------------------------------- #
def test_all_conflicts_considered_beyond_default_limit(session) -> None:
    feature = _feature_with_n_conflicts(session, 101)
    persisted = {c.id for c in ConflictRepository(session).list_by_subjects([feature.id])}
    assert len(persisted) == 101

    result = _decision_service(session).synthesize_decisions([feature.id])
    decision = result.decisions[0]

    # Every one of the 101 conflicts was considered, required, and acknowledged.
    acknowledged = {edge.conflict_id for edge in decision.acknowledged_conflicts}
    assert len(acknowledged) == 101
    assert acknowledged == persisted


# --------------------------------------------------------------------------- #
# B) Completeness still fails when any required conflict is omitted, at scale
# --------------------------------------------------------------------------- #
def test_gate_fails_when_one_of_many_conflicts_omitted(model_meta) -> None:
    """101 conflicts over a subject; a decision acknowledging only 100 is rejected."""

    subject = uuid.uuid4()
    signal = uuid.uuid4()
    conflict_ids = [uuid.uuid4() for _ in range(101)]
    conflict_subjects = {cid: subject for cid in conflict_ids}

    decision = DecisionContract(
        decision_id="d1",
        subject_type="feature",
        subject_id=subject,
        recommendation="build_now",
        title="t",
        rationale="acknowledges most but not all conflicts over the subject",
        priority_rank=1,
        acknowledged_conflict_ids=conflict_ids[:100],  # omit exactly the 101st
        evidence_signal_ids=[signal],
        confidence={"score": 0.8},
    )
    synthesis = DecisionSynthesisContract(decisions=[decision], model_meta=model_meta)

    report = validate_decision_integrity(synthesis, [subject], [signal], conflict_subjects)
    assert report.passed is False
    assert any(cid == conflict_ids[100] for _, cid in report.unacknowledged_conflicts)


def test_synthesis_fails_when_conflicts_ignored_at_scale(session) -> None:
    """End-to-end: with 101 conflicts, a decision that ignores them cannot be persisted."""

    feature = _feature_with_n_conflicts(session, 101)
    service = _decision_service(session, mode="ignore_conflict", max_attempts=2)
    with pytest.raises(DecisionSynthesisFailedError):
        service.synthesize_decisions([feature.id])
    assert service.list_decisions() == []


# --------------------------------------------------------------------------- #
# C) No regression for normal conflict counts
# --------------------------------------------------------------------------- #
def test_no_regression_for_normal_conflict_count(session) -> None:
    feature = _feature_with_n_conflicts(session, 3)
    decision = _decision_service(session).synthesize_decisions([feature.id]).decisions[0]
    assert len(decision.acknowledged_conflicts) == 3
