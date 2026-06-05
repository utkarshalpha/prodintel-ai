"""Phase 5 (C + E) -- DecisionExplanationService end-to-end + traceability.

Drives the real Stage 1→2→3→4 pipeline with deterministic fakes, then explains the
resulting decision and asserts the full provenance chain -- including the headline
traceability invariant: ``quoted_text == raw_text[source_span.start:end]``.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.repositories.conflict_repository import ConflictRepository
from app.repositories.decision_repository import DecisionRepository
from app.repositories.feature_repository import FeatureRepository
from app.repositories.signal_repository import SignalRepository
from app.services.decision_explanation_service import DecisionExplanationService
from app.services.errors import DecisionNotFoundError

from tests.app_helpers import make_engine, make_session_factory
from tests.test_decision_service import _decision_service, _feature_with_conflict


@pytest.fixture
def session():
    db = make_session_factory(make_engine())()
    try:
        yield db
    finally:
        db.close()


def _service(session) -> DecisionExplanationService:
    return DecisionExplanationService(
        decision_repo=DecisionRepository(session),
        feature_repo=FeatureRepository(session),
        conflict_repo=ConflictRepository(session),
        signal_repo=SignalRepository(session),
    )


def _build_decision(session):
    feature_id = _feature_with_conflict(session)
    decision = _decision_service(session).synthesize_decisions([feature_id]).decisions[0]
    return decision, feature_id


def test_explain_returns_full_provenance(session) -> None:
    decision, feature_id = _build_decision(session)

    exp = _service(session).explain(decision.id)

    assert exp.decision.id == decision.id
    assert exp.subject_feature is not None and exp.subject_feature.id == feature_id
    assert exp.direct_evidence, "decision should expose direct evidence signals"
    assert exp.conflicts and exp.conflicts[0].parties, "acknowledged conflicts + parties present"
    assert exp.signals, "original stakeholder inputs present"
    assert all(sp.raw_text for sp in exp.signals), "raw stakeholder text is reachable"
    assert exp.integrity.complete is True
    # Sales/Engineering signals are reached via direct evidence + feature + conflict.
    assert any(len(sp.reached_via) == 3 for sp in exp.signals)
    assert exp.meta.decision_id == decision.id


def test_explain_missing_decision_raises(session) -> None:
    with pytest.raises(DecisionNotFoundError):
        _service(session).explain(uuid4())


def test_traceability_quoted_text_matches_raw_text(session) -> None:
    """(E) Every claim's quoted_text is exactly the raw_text substring at its span."""

    decision, _ = _build_decision(session)
    exp = _service(session).explain(decision.id)

    checked = 0
    for sp in exp.signals:
        if sp.analysis is None:
            continue
        for claim in sp.analysis.claims:
            start, end = claim.source_span
            assert claim.quoted_text == sp.raw_text[start:end]
            checked += 1
    assert checked > 0, "expected at least one grounded claim to verify"
