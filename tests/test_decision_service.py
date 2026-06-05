"""Business-capability tests for DecisionService (Stage 1 -> 2 -> 3 -> 4 end-to-end)."""

from __future__ import annotations

import uuid

import pytest

from app.ai_contracts.enums import StakeholderType
from app.ai_runtime.retry_policy import RetryPolicy
from app.repositories.conflict_repository import ConflictRepository
from app.repositories.decision_repository import DecisionRepository
from app.repositories.feature_repository import FeatureRepository, FeatureSignalRepository
from app.repositories.parsed_signal_repository import ParsedSignalRepository
from app.repositories.signal_repository import SignalRepository
from app.services.conflict_service import ConflictService
from app.services.decision_service import DecisionService
from app.services.errors import (
    DecisionNotFoundError,
    DecisionSynthesisFailedError,
    FeatureNotFoundError,
)
from app.services.feature_service import FeatureService
from app.services.signal_service import SignalService
from app.stages.stage1.runner import build_stage1_runner
from app.stages.stage2.runner import build_stage2_runner
from app.stages.stage3.runner import build_stage3_runner
from app.stages.stage4.runner import build_stage4_runner

from tests.app_helpers import (
    Stage1FakeClient,
    Stage2FakeClient,
    Stage3FakeClient,
    Stage4FakeClient,
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


def _conflict_service(session) -> ConflictService:
    return ConflictService(
        session, ConflictRepository(session), FeatureRepository(session),
        ParsedSignalRepository(session),
        build_stage3_runner(Stage3FakeClient(mode="conflict")),
    )


def _decision_service(session, *, mode: str = "recommend", max_attempts: int = 3) -> DecisionService:
    return DecisionService(
        session, DecisionRepository(session), ConflictRepository(session),
        FeatureRepository(session), ParsedSignalRepository(session),
        build_stage4_runner(Stage4FakeClient(mode=mode), retry_policy=RetryPolicy(max_attempts=max_attempts)),
    )


def _feature_with_conflict(session) -> uuid.UUID:
    """Build a feature backed by Sales + Engineering signals and detect a conflict over it."""

    signals = _signal_service(session)
    sales = signals.create_signal(source_type=StakeholderType.SALES, raw_text="Enterprise SSO is critical").signal
    eng = signals.create_signal(source_type=StakeholderType.ENGINEERING, raw_text="SSO is high risk to build").signal
    signals.analyze_signal(sales.id)
    signals.analyze_signal(eng.id)
    feature = _feature_service(session).extract_features([sales.id, eng.id]).features[0]
    _conflict_service(session).detect_conflicts([feature.id])
    return feature.id


# --------------------------------------------------------------------------- #
# synthesize_decisions
# --------------------------------------------------------------------------- #
def test_synthesize_persists_evidence_traceable_decision(session) -> None:
    feature_id = _feature_with_conflict(session)

    result = _decision_service(session).synthesize_decisions([feature_id])

    assert len(result.decisions) == 1
    decision = result.decisions[0]
    assert decision.subject_id == feature_id
    # evidence-traceable: signals backing it and the conflict it acknowledges are persisted
    assert decision.evidence, "decision must cite evidence signals"
    assert decision.acknowledged_conflicts, "decision must acknowledge the conflict over its subject"
    # the acknowledged conflict is the real persisted conflict over this feature
    conflict_ids = {edge.conflict_id for edge in decision.acknowledged_conflicts}
    persisted = {c.id for c in _conflict_service(session).list_conflicts(subject_id=feature_id)}
    assert conflict_ids == persisted
    assert _decision_service(session).get_decision(decision.id).id == decision.id


def test_synthesize_failure_persists_nothing(session) -> None:
    feature_id = _feature_with_conflict(session)
    service = _decision_service(session, mode="ignore_conflict", max_attempts=2)
    with pytest.raises(DecisionSynthesisFailedError):
        service.synthesize_decisions([feature_id])
    assert service.list_decisions() == []


def test_synthesize_missing_feature_raises(session) -> None:
    with pytest.raises(FeatureNotFoundError):
        _decision_service(session).synthesize_decisions([uuid.uuid4()])


# --------------------------------------------------------------------------- #
# get / list
# --------------------------------------------------------------------------- #
def test_list_by_subject(session) -> None:
    feature_id = _feature_with_conflict(session)
    _decision_service(session).synthesize_decisions([feature_id])
    listed = _decision_service(session).list_decisions(subject_id=feature_id)
    assert len(listed) == 1
    assert listed[0].subject_id == feature_id


def test_get_missing_decision_raises(session) -> None:
    with pytest.raises(DecisionNotFoundError):
        _decision_service(session).get_decision(uuid.uuid4())
