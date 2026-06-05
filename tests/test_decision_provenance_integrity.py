"""Stage 4 provenance integrity tests: deletion protection + edge cleanup.

Proves the guarantee claimed by ADR-012, the README, and PROJECT_STATE — *"a signal
or conflict backing a decision cannot be deleted out from under it"* — which the
Stage 4 review flagged as untested (HIGH #1).

Mirrors the Stage 2 reference suite (``tests/test_provenance_integrity.py``): rows are
built directly through the ORM against a SQLite engine with ``PRAGMA foreign_keys=ON``
(see ``make_engine``), so ``ON DELETE RESTRICT`` / ``CASCADE`` behave as they do on
production Postgres.

Coverage:
* A — deleting a signal backing a decision (via ``decision_evidence``) is rejected.
* B — deleting a conflict acknowledged by a decision (via ``decision_conflict``) is
  rejected.
* C — deleting the decision cascades both edge sets, leaves the signal and conflict
  intact, and leaves no orphan edge rows.
* D — the same guarantee holds for edges produced by the real Stage 1→2→3→4 service
  pipeline (service-level coverage), using deterministic fakes (no network).
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.ai_contracts.enums import (
    ConflictType,
    DecisionRecommendation,
    StakeholderType,
    SubjectType,
)
from app.models.conflict import Conflict
from app.models.decision import Decision, DecisionConflict, DecisionEvidence
from app.models.signal import Signal

from tests.app_helpers import make_engine, make_session_factory
from tests.test_decision_service import _decision_service, _feature_with_conflict


@pytest.fixture
def session():
    db = make_session_factory(make_engine())()
    try:
        yield db
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# Builders (reference style: minimal, valid rows with overridable defaults)
# --------------------------------------------------------------------------- #
def _signal(session, text: str = "a backing signal") -> Signal:
    signal = Signal(source_type=StakeholderType.CUSTOMER, raw_text=text, content_hash=uuid.uuid4().hex)
    session.add(signal)
    session.flush()
    return signal


def _conflict(session) -> Conflict:
    conflict = Conflict(
        subject_type=SubjectType.FEATURE,
        subject_id=uuid.uuid4(),
        conflict_type=ConflictType.RISK,
        severity=3,
        confidence_score=0.8,
        confidence={"score": 0.8, "basis": "strong", "components": {}},
        model_meta={"source": "test"},
    )
    session.add(conflict)
    session.flush()
    return conflict


def _decision(session) -> Decision:
    return Decision(
        subject_type=SubjectType.FEATURE,
        subject_id=uuid.uuid4(),  # polymorphic, no FK — a bare uuid is valid here
        recommendation=DecisionRecommendation.BUILD_NOW,
        title="Build it",
        rationale="the evidence supports building it",
        priority_rank=1,
        confidence_score=0.8,
        confidence={"score": 0.8, "basis": "strong", "components": {}},
        model_meta={"source": "test"},
    )


# --------------------------------------------------------------------------- #
# A — signal deletion protection (decision_evidence -> signal, RESTRICT)
# --------------------------------------------------------------------------- #
def test_signal_backing_a_decision_cannot_be_deleted(session) -> None:
    signal = _signal(session)
    decision = _decision(session)
    decision.evidence.append(DecisionEvidence(signal_id=signal.id))
    session.add(decision)
    session.commit()

    session.delete(signal)
    with pytest.raises(IntegrityError):
        session.flush()
    session.rollback()

    # The signal and its edge survive the rejected delete.
    assert session.get(Signal, signal.id) is not None
    assert session.get(DecisionEvidence, {"decision_id": decision.id, "signal_id": signal.id}) is not None


# --------------------------------------------------------------------------- #
# B — conflict deletion protection (decision_conflict -> conflict, RESTRICT)
# --------------------------------------------------------------------------- #
def test_conflict_backing_a_decision_cannot_be_deleted(session) -> None:
    conflict = _conflict(session)
    decision = _decision(session)
    decision.acknowledged_conflicts.append(DecisionConflict(conflict_id=conflict.id))
    session.add(decision)
    session.commit()

    session.delete(conflict)
    with pytest.raises(IntegrityError):
        session.flush()
    session.rollback()

    # The conflict and its edge survive the rejected delete.
    assert session.get(Conflict, conflict.id) is not None
    assert session.get(DecisionConflict, {"decision_id": decision.id, "conflict_id": conflict.id}) is not None


# --------------------------------------------------------------------------- #
# C — deleting a decision cascades its edges, leaving no orphans
# --------------------------------------------------------------------------- #
def test_deleting_decision_cascades_its_provenance_edges(session) -> None:
    signal = _signal(session)
    conflict = _conflict(session)
    decision = _decision(session)
    decision.evidence.append(DecisionEvidence(signal_id=signal.id))
    decision.acknowledged_conflicts.append(DecisionConflict(conflict_id=conflict.id))
    session.add(decision)
    session.commit()

    session.delete(decision)
    session.commit()

    # Both edges are gone (CASCADE from the decision side)...
    assert session.get(DecisionEvidence, {"decision_id": decision.id, "signal_id": signal.id}) is None
    assert session.get(DecisionConflict, {"decision_id": decision.id, "conflict_id": conflict.id}) is None
    # ...the upstream signal and conflict remain (RESTRICT only blocks deleting *them*)...
    assert session.get(Signal, signal.id) is not None
    assert session.get(Conflict, conflict.id) is not None
    # ...and no orphan edge rows are left behind anywhere.
    assert session.scalars(select(DecisionEvidence)).all() == []
    assert session.scalars(select(DecisionConflict)).all() == []


# --------------------------------------------------------------------------- #
# D — the guarantee holds for edges produced by the real service pipeline
# --------------------------------------------------------------------------- #
def test_service_produced_decision_edges_are_fk_protected(session) -> None:
    """Stage 1→2→3→4 with deterministic fakes; the persisted edges are FK-protected."""

    feature_id = _feature_with_conflict(session)
    decision = _decision_service(session).synthesize_decisions([feature_id]).decisions[0]

    # The service actually writes FK-backed provenance edges (the reason the
    # guarantee matters): real evidence signals and a real acknowledged conflict.
    assert decision.evidence, "decision must persist evidence edges"
    assert decision.acknowledged_conflicts, "decision must persist acknowledged-conflict edges"
    signal_id = decision.evidence[0].signal_id
    conflict_id = decision.acknowledged_conflicts[0].conflict_id

    # A backing signal cannot be deleted out from under the decision.
    session.delete(session.get(Signal, signal_id))
    with pytest.raises(IntegrityError):
        session.flush()
    session.rollback()

    # Nor can an acknowledged conflict.
    session.delete(session.get(Conflict, conflict_id))
    with pytest.raises(IntegrityError):
        session.flush()
    session.rollback()

    assert session.get(Signal, signal_id) is not None
    assert session.get(Conflict, conflict_id) is not None
