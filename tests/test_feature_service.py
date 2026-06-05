"""Business-capability tests for FeatureService (Stage 1 -> Stage 2 end-to-end)."""

from __future__ import annotations

import uuid

import pytest

from app.ai_contracts.enums import StakeholderType
from app.ai_runtime.retry_policy import RetryPolicy
from app.repositories.feature_repository import FeatureRepository, FeatureSignalRepository
from app.repositories.parsed_signal_repository import ParsedSignalRepository
from app.repositories.signal_repository import SignalRepository
from app.services.errors import (
    FeatureExtractionFailedError,
    FeatureNotFoundError,
    SignalsNotAnalyzedError,
)
from app.services.feature_service import FeatureService
from app.services.signal_service import SignalService
from app.stages.stage1.runner import build_stage1_runner
from app.stages.stage2.runner import build_stage2_runner

from tests.app_helpers import Stage1FakeClient, Stage2FakeClient, make_engine, make_session_factory


@pytest.fixture
def session():
    engine = make_engine()
    db = make_session_factory(engine)()
    try:
        yield db
    finally:
        db.close()


def _signal_service(session) -> SignalService:
    return SignalService(
        session,
        SignalRepository(session),
        ParsedSignalRepository(session),
        build_stage1_runner(Stage1FakeClient(grounded=True)),
    )


def _feature_service(session, *, mode: str = "cluster_all", max_attempts: int = 3) -> FeatureService:
    return FeatureService(
        session,
        FeatureRepository(session),
        FeatureSignalRepository(session),
        ParsedSignalRepository(session),
        SignalRepository(session),
        build_stage2_runner(Stage2FakeClient(mode=mode), retry_policy=RetryPolicy(max_attempts=max_attempts)),
    )


def _analyzed_signal(signal_service: SignalService, text: str) -> uuid.UUID:
    signal = signal_service.create_signal(source_type=StakeholderType.CUSTOMER, raw_text=text).signal
    signal_service.analyze_signal(signal.id)
    return signal.id


# --------------------------------------------------------------------------- #
# extract_features
# --------------------------------------------------------------------------- #
def test_extract_persists_features_with_provenance(session) -> None:
    signals = _signal_service(session)
    s1 = _analyzed_signal(signals, "Mobile checkout fails on pay")
    s2 = _analyzed_signal(signals, "Customers cannot complete payment on phones")

    features = _feature_service(session)
    result = features.extract_features([s1, s2])

    assert len(result.features) == 1
    feature = result.features[0]
    assert feature.id is not None
    assert {link.signal_id for link in feature.feature_signals} == {s1, s2}
    # retrievable + listable by signal
    assert features.get_feature(feature.id).id == feature.id
    assert feature.id in {f.id for f in features.list_features(signal_id=s1)}


def test_extract_requires_analyzed_signals(session) -> None:
    signals = _signal_service(session)
    unanalyzed = signals.create_signal(source_type=StakeholderType.SALES, raw_text="not analyzed yet").signal

    features = _feature_service(session)
    with pytest.raises(SignalsNotAnalyzedError):
        features.extract_features([unanalyzed.id])


def test_extract_failure_persists_nothing(session) -> None:
    signals = _signal_service(session)
    s1 = _analyzed_signal(signals, "Some analyzed signal text")

    features = _feature_service(session, mode="fabricate", max_attempts=2)
    with pytest.raises(FeatureExtractionFailedError):
        features.extract_features([s1])

    assert features.list_features() == []


# --------------------------------------------------------------------------- #
# create_feature / link
# --------------------------------------------------------------------------- #
def test_manual_create_feature(session) -> None:
    signals = _signal_service(session)
    s1 = _analyzed_signal(signals, "Signal backing a manual feature")

    features = _feature_service(session)
    feature = features.create_feature(
        title="Manual feature",
        description="Created by hand",
        jtbd="When x, I want y, so I can z",
        source_signal_ids=[s1],
    )
    assert feature.model_meta["source"] == "manual"
    assert {link.signal_id for link in feature.feature_signals} == {s1}


def test_link_feature_to_signal_is_idempotent(session) -> None:
    signals = _signal_service(session)
    s1 = _analyzed_signal(signals, "First signal")
    s2 = _analyzed_signal(signals, "Second signal")

    features = _feature_service(session)
    feature = features.create_feature(
        title="Feature", description="d", jtbd="When x, I want y, so I can z", source_signal_ids=[s1]
    )
    first = features.link_feature_to_signal(feature.id, s2)
    again = features.link_feature_to_signal(feature.id, s2)
    assert (first.feature_id, first.signal_id) == (again.feature_id, again.signal_id)


def test_get_missing_feature_raises(session) -> None:
    features = _feature_service(session)
    with pytest.raises(FeatureNotFoundError):
        features.get_feature(uuid.uuid4())
