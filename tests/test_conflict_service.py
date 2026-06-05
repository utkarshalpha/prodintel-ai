"""Business-capability tests for ConflictService (Stage 1 -> 2 -> 3 end-to-end)."""

from __future__ import annotations

import uuid

import pytest

from app.ai_contracts.enums import StakeholderType
from app.ai_runtime.retry_policy import RetryPolicy
from app.repositories.conflict_repository import ConflictRepository
from app.repositories.feature_repository import FeatureRepository, FeatureSignalRepository
from app.repositories.parsed_signal_repository import ParsedSignalRepository
from app.repositories.signal_repository import SignalRepository
from app.services.conflict_service import ConflictService
from app.services.errors import (
    ConflictDetectionFailedError,
    ConflictNotFoundError,
    FeatureNotFoundError,
)
from app.services.feature_service import FeatureService
from app.services.signal_service import SignalService
from app.stages.stage1.runner import build_stage1_runner
from app.stages.stage2.runner import build_stage2_runner
from app.stages.stage3.runner import build_stage3_runner

from tests.app_helpers import (
    Stage1FakeClient,
    Stage2FakeClient,
    Stage3FakeClient,
    make_engine,
    make_session_factory,
)


@pytest.fixture
def session():
    db = make_session_factory(make_engine())()
    try:
        yield db
    finally:
        db.close()


def _signal_service(session) -> SignalService:
    return SignalService(
        session, SignalRepository(session), ParsedSignalRepository(session),
        build_stage1_runner(Stage1FakeClient(grounded=True)),
    )


def _feature_service(session) -> FeatureService:
    return FeatureService(
        session, FeatureRepository(session), FeatureSignalRepository(session),
        ParsedSignalRepository(session), SignalRepository(session),
        build_stage2_runner(Stage2FakeClient(mode="cluster_all")),
    )


def _conflict_service(session, *, mode: str = "conflict", max_attempts: int = 3) -> ConflictService:
    return ConflictService(
        session, ConflictRepository(session), FeatureRepository(session),
        ParsedSignalRepository(session),
        build_stage3_runner(Stage3FakeClient(mode=mode), retry_policy=RetryPolicy(max_attempts=max_attempts)),
    )


def _feature_over_two_stakeholders(session) -> uuid.UUID:
    """Create a feature backed by a Sales signal and an Engineering signal."""

    signals = _signal_service(session)
    sales = signals.create_signal(source_type=StakeholderType.SALES, raw_text="Enterprise SSO is critical").signal
    eng = signals.create_signal(source_type=StakeholderType.ENGINEERING, raw_text="SSO is high risk to build").signal
    signals.analyze_signal(sales.id)
    signals.analyze_signal(eng.id)
    feature = _feature_service(session).extract_features([sales.id, eng.id]).features[0]
    return feature.id


# --------------------------------------------------------------------------- #
# detect_conflicts
# --------------------------------------------------------------------------- #
def test_detect_persists_evidence_traceable_conflict(session) -> None:
    feature_id = _feature_over_two_stakeholders(session)

    result = _conflict_service(session).detect_conflicts([feature_id])

    assert len(result.conflicts) == 1
    conflict = result.conflicts[0]
    assert conflict.subject_id == feature_id
    assert {p.stakeholder_type for p in conflict.parties} == {StakeholderType.SALES, StakeholderType.ENGINEERING}
    # every position is evidence-traceable
    for party in conflict.parties:
        assert party.evidence_signal_ids
    assert _conflict_service(session).get_conflict(conflict.id).id == conflict.id


def test_detect_no_conflict_persists_nothing(session) -> None:
    feature_id = _feature_over_two_stakeholders(session)
    result = _conflict_service(session, mode="none").detect_conflicts([feature_id])
    assert result.conflicts == []
    assert _conflict_service(session).list_conflicts() == []


def test_detect_failure_persists_nothing(session) -> None:
    feature_id = _feature_over_two_stakeholders(session)
    service = _conflict_service(session, mode="fabricate", max_attempts=2)
    with pytest.raises(ConflictDetectionFailedError):
        service.detect_conflicts([feature_id])
    assert service.list_conflicts() == []


def test_detect_missing_feature_raises(session) -> None:
    with pytest.raises(FeatureNotFoundError):
        _conflict_service(session).detect_conflicts([uuid.uuid4()])


# --------------------------------------------------------------------------- #
# get / list
# --------------------------------------------------------------------------- #
def test_list_by_subject(session) -> None:
    feature_id = _feature_over_two_stakeholders(session)
    _conflict_service(session).detect_conflicts([feature_id])
    listed = _conflict_service(session).list_conflicts(subject_id=feature_id)
    assert len(listed) == 1
    assert listed[0].subject_id == feature_id


def test_get_missing_conflict_raises(session) -> None:
    with pytest.raises(ConflictNotFoundError):
        _conflict_service(session).get_conflict(uuid.uuid4())
