"""Phase 5 (B) -- repository tests for the explanation traversal.

Verifies eager loading (no N+1), deterministic ordering, and provenance traversal of
the four read methods: ``get_with_provenance``, ``get_with_signals``,
``get_many_with_parties``, ``list_with_analysis``.
"""

from __future__ import annotations

from contextlib import contextmanager
from uuid import uuid4

import pytest
from sqlalchemy import event

from app.repositories.conflict_repository import ConflictRepository
from app.repositories.decision_repository import DecisionRepository
from app.repositories.feature_repository import FeatureRepository
from app.repositories.signal_repository import SignalRepository

from tests.app_helpers import make_engine, make_session_factory
from tests.test_decision_service import _decision_service, _feature_with_conflict


@pytest.fixture
def session():
    db = make_session_factory(make_engine())()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def count_queries(session):
    """Count SQL statements executed on the session's bind within the block."""

    counter = {"n": 0}
    bind = session.get_bind()

    def _inc(*_args, **_kwargs):
        counter["n"] += 1

    event.listen(bind, "after_cursor_execute", _inc)
    try:
        yield counter
    finally:
        event.remove(bind, "after_cursor_execute", _inc)


def _build_decision(session):
    feature_id = _feature_with_conflict(session)
    decision = _decision_service(session).synthesize_decisions([feature_id]).decisions[0]
    return decision, feature_id


# --------------------------------------------------------------------------- #
# DecisionRepository.get_with_provenance
# --------------------------------------------------------------------------- #
def test_get_with_provenance_eager_loads_edges(session) -> None:
    decision, _ = _build_decision(session)
    session.expunge_all()

    loaded = DecisionRepository(session).get_with_provenance(decision.id)
    assert loaded is not None
    with count_queries(session) as c:
        _ = [e.signal_id for e in loaded.evidence]
        _ = [e.conflict_id for e in loaded.acknowledged_conflicts]
    assert c["n"] == 0  # no N+1: edges were eager-loaded


def test_get_with_provenance_missing_returns_none(session) -> None:
    assert DecisionRepository(session).get_with_provenance(uuid4()) is None


# --------------------------------------------------------------------------- #
# FeatureRepository.get_with_signals
# --------------------------------------------------------------------------- #
def test_get_with_signals_eager_loads(session) -> None:
    _, feature_id = _build_decision(session)
    session.expunge_all()

    feature = FeatureRepository(session).get_with_signals(feature_id)
    assert feature is not None
    with count_queries(session) as c:
        ids = [fs.signal_id for fs in feature.feature_signals]
    assert c["n"] == 0
    assert len(ids) >= 1
    assert FeatureRepository(session).get_with_signals(uuid4()) is None


# --------------------------------------------------------------------------- #
# ConflictRepository.get_many_with_parties
# --------------------------------------------------------------------------- #
def test_get_many_with_parties_ordered_and_eager(session) -> None:
    decision, _ = _build_decision(session)
    conflict_ids = [e.conflict_id for e in decision.acknowledged_conflicts]
    session.expunge_all()

    repo = ConflictRepository(session)
    conflicts = list(repo.get_many_with_parties(conflict_ids))
    assert {c.id for c in conflicts} == set(conflict_ids)
    severities = [c.severity for c in conflicts]
    assert severities == sorted(severities, reverse=True)  # deterministic order
    with count_queries(session) as c:
        for conflict in conflicts:
            _ = list(conflict.parties)
    assert c["n"] == 0
    assert repo.get_many_with_parties([]) == []


# --------------------------------------------------------------------------- #
# SignalRepository.list_with_analysis
# --------------------------------------------------------------------------- #
def test_list_with_analysis_ordered_and_eager(session) -> None:
    _, feature_id = _build_decision(session)
    signal_ids = [fs.signal_id for fs in FeatureRepository(session).get_with_signals(feature_id).feature_signals]
    session.expunge_all()

    repo = SignalRepository(session)
    signals = list(repo.list_with_analysis(signal_ids))
    assert {s.id for s in signals} == set(signal_ids)
    assert [s.id for s in signals] == sorted(s.id for s in signals)  # ordered by id
    with count_queries(session) as c:
        for s in signals:
            _ = s.parsed.intent if s.parsed else None
    assert c["n"] == 0
    assert repo.list_with_analysis([]) == []
