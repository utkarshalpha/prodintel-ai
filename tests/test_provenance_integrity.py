"""Provenance integrity tests: deletion protection, reverse traversal, index usage.

Covers the data-integrity guarantees behind the feature_signal provenance edge:
* a signal backing a feature cannot be deleted (ON DELETE RESTRICT),
* features are reachable in reverse from a signal, and
* the reverse lookup uses the ix_feature_signal_signal_id index (HIGH #1).
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.exc import IntegrityError

from app.ai_contracts.enums import StakeholderType
from app.models.feature import Feature, FeatureSignal
from app.models.signal import Signal
from app.repositories.feature_repository import FeatureRepository

from tests.app_helpers import make_engine, make_session_factory


def _signal(session, text: str = "a signal") -> Signal:
    signal = Signal(source_type=StakeholderType.CUSTOMER, raw_text=text, content_hash=uuid.uuid4().hex)
    session.add(signal)
    session.flush()
    return signal


def _feature_linked_to(session, signal: Signal) -> Feature:
    feature = Feature(
        title="Feature",
        description="d",
        jtbd="When x, I want y, so I can z",
        confidence_score=1.0,
        confidence={"score": 1.0, "basis": "strong", "components": {}},
        model_meta={"source": "test"},
    )
    feature.feature_signals.append(FeatureSignal(signal_id=signal.id))
    session.add(feature)
    session.commit()
    return feature


@pytest.fixture
def session():
    db = make_session_factory(make_engine())()
    try:
        yield db
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# Deletion protection (ON DELETE RESTRICT)
# --------------------------------------------------------------------------- #
def test_signal_backing_a_feature_cannot_be_deleted(session) -> None:
    signal = _signal(session)
    _feature_linked_to(session, signal)

    session.delete(signal)
    with pytest.raises(IntegrityError):
        session.flush()
    session.rollback()


def test_deleting_feature_cascades_its_provenance_edges(session) -> None:
    signal = _signal(session)
    feature = _feature_linked_to(session, signal)

    session.delete(feature)
    session.commit()

    # The provenance edge is gone; the signal remains (RESTRICT only blocks signal).
    assert session.get(FeatureSignal, {"feature_id": feature.id, "signal_id": signal.id}) is None
    assert session.get(Signal, signal.id) is not None


# --------------------------------------------------------------------------- #
# Reverse traversal
# --------------------------------------------------------------------------- #
def test_reverse_traversal_finds_features_for_signal(session) -> None:
    signal = _signal(session)
    feature = _feature_linked_to(session, signal)

    found = FeatureRepository(session).list(signal_id=signal.id)
    assert [f.id for f in found] == [feature.id]


# --------------------------------------------------------------------------- #
# Index usage (HIGH #1 verification)
# --------------------------------------------------------------------------- #
def test_reverse_lookup_uses_signal_id_index() -> None:
    engine = make_engine()
    with engine.connect() as conn:
        rows = conn.exec_driver_sql(
            "EXPLAIN QUERY PLAN SELECT * FROM feature_signal WHERE signal_id = '00000000000000000000000000000000'"
        ).fetchall()
    plan = " ".join(str(row) for row in rows)
    assert "ix_feature_signal_signal_id" in plan
    engine.dispose()
