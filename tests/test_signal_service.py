"""Unit/integration tests for SignalService and the repositories.

Uses real repositories over an in-memory SQLite database plus a fake Stage 1 client,
so the full application path (hash/dedupe, persistence, analysis, grounding) is
exercised without any network.
"""

from __future__ import annotations

import uuid

import pytest

from app.ai_contracts.enums import StakeholderType
from app.ai_runtime.retry_policy import RetryPolicy
from app.models.signal import SignalImmutableError
from app.repositories.parsed_signal_repository import ParsedSignalRepository
from app.repositories.signal_repository import SignalRepository
from app.services.errors import (
    AnalysisFailedError,
    AnalysisNotFoundError,
    SignalNotFoundError,
)
from app.services.signal_service import SignalService
from app.stages.stage1.runner import build_stage1_runner

from tests.app_helpers import Stage1FakeClient, make_engine, make_session_factory


def _service(session, *, grounded: bool = True, max_attempts: int = 3) -> SignalService:
    runner = build_stage1_runner(
        Stage1FakeClient(grounded=grounded),
        retry_policy=RetryPolicy(max_attempts=max_attempts),
    )
    return SignalService(session, SignalRepository(session), ParsedSignalRepository(session), runner)


@pytest.fixture
def session():
    engine = make_engine()
    factory = make_session_factory(engine)
    db = factory()
    try:
        yield db
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# create / dedupe
# --------------------------------------------------------------------------- #
def test_create_signal_persists_and_hashes(session) -> None:
    service = _service(session)
    result = service.create_signal(source_type=StakeholderType.CUSTOMER, raw_text="Checkout is broken")

    assert result.created is True
    assert result.signal.id is not None
    assert len(result.signal.content_hash) == 64  # sha-256 hex
    assert result.signal.created_at is not None


def test_create_signal_is_idempotent_on_duplicate(session) -> None:
    service = _service(session)
    first = service.create_signal(source_type=StakeholderType.SALES, raw_text="Same text")
    second = service.create_signal(source_type=StakeholderType.SALES, raw_text="Same text")

    assert first.created is True
    assert second.created is False
    assert second.signal.id == first.signal.id


# --------------------------------------------------------------------------- #
# get
# --------------------------------------------------------------------------- #
def test_get_signal_found(session) -> None:
    service = _service(session)
    created = service.create_signal(source_type=StakeholderType.SUPPORT, raw_text="A ticket").signal
    assert service.get_signal(created.id).id == created.id


def test_get_signal_missing_raises(session) -> None:
    service = _service(session)
    with pytest.raises(SignalNotFoundError):
        service.get_signal(uuid.uuid4())


# --------------------------------------------------------------------------- #
# analyze
# --------------------------------------------------------------------------- #
def test_analyze_persists_grounded_result(session) -> None:
    service = _service(session, grounded=True)
    signal = service.create_signal(source_type=StakeholderType.CUSTOMER, raw_text="Mobile checkout fails on pay").signal

    result = service.analyze_signal(signal.id)

    assert result.parsed_signal.signal_id == signal.id
    assert result.parsed_signal.intent
    assert result.parsed_signal.model_meta["model_id"] == "claude-test"
    # And it is retrievable.
    fetched = service.get_analysis(signal.id)
    assert fetched.id == result.parsed_signal.id


def test_analyze_failure_persists_nothing(session) -> None:
    service = _service(session, grounded=False, max_attempts=2)
    signal = service.create_signal(source_type=StakeholderType.CUSTOMER, raw_text="Some signal text here").signal

    with pytest.raises(AnalysisFailedError):
        service.analyze_signal(signal.id)

    with pytest.raises(AnalysisNotFoundError):
        service.get_analysis(signal.id)


def test_reanalyze_replaces_previous_analysis(session) -> None:
    service = _service(session, grounded=True)
    signal = service.create_signal(source_type=StakeholderType.CUSTOMER, raw_text="Analyze me twice please").signal

    first = service.analyze_signal(signal.id).parsed_signal
    second = service.analyze_signal(signal.id).parsed_signal

    # Still exactly one analysis row for this signal.
    parsed_repo = ParsedSignalRepository(session)
    assert parsed_repo.get_by_signal_id(signal.id).id == second.id
    assert first.id != second.id  # the old row was replaced


def test_analyze_missing_signal_raises(session) -> None:
    service = _service(session)
    with pytest.raises(SignalNotFoundError):
        service.analyze_signal(uuid.uuid4())


# --------------------------------------------------------------------------- #
# immutability
# --------------------------------------------------------------------------- #
def test_signal_is_immutable(session) -> None:
    service = _service(session)
    signal = service.create_signal(source_type=StakeholderType.CUSTOMER, raw_text="Do not mutate me").signal

    signal.raw_text = "tampered"
    with pytest.raises(SignalImmutableError):
        session.flush()
