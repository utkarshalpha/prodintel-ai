"""End-to-end API tests for the signal endpoints via FastAPI TestClient.

Overrides the DB dependency with an in-memory engine and injects a fake Stage 1
runner on ``app.state``, so the HTTP surface is tested with no network.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from app.ai_runtime.retry_policy import RetryPolicy
from app.api.app import create_app
from app.api.deps import get_db
from app.stages.stage1.runner import build_stage1_runner

from tests.app_helpers import Stage1FakeClient, make_engine, make_session_factory


def _build_client(*, grounded: bool = True, max_attempts: int = 3) -> TestClient:
    engine = make_engine()
    factory = make_session_factory(engine)

    def override_get_db():
        db = factory()
        try:
            yield db
        finally:
            db.close()

    app = create_app()
    app.dependency_overrides[get_db] = override_get_db
    app.state.stage1_runner = build_stage1_runner(
        Stage1FakeClient(grounded=grounded),
        retry_policy=RetryPolicy(max_attempts=max_attempts),
    )
    return TestClient(app)


@pytest.fixture
def client() -> TestClient:
    return _build_client()


def _create(client: TestClient, text: str = "Mobile checkout fails on pay") -> dict:
    resp = client.post("/signals", json={"source_type": "customer", "raw_text": text})
    assert resp.status_code == 201, resp.text
    return resp.json()


# --------------------------------------------------------------------------- #
# create
# --------------------------------------------------------------------------- #
def test_create_signal_returns_201(client: TestClient) -> None:
    body = _create(client)
    assert body["source_type"] == "customer"
    assert len(body["content_hash"]) == 64
    assert uuid.UUID(body["id"])


def test_duplicate_create_returns_200(client: TestClient) -> None:
    first = _create(client, "duplicate text")
    resp = client.post("/signals", json={"source_type": "customer", "raw_text": "duplicate text"})
    assert resp.status_code == 200
    assert resp.json()["id"] == first["id"]


def test_create_rejects_unknown_stakeholder(client: TestClient) -> None:
    resp = client.post("/signals", json={"source_type": "marketing", "raw_text": "x"})
    assert resp.status_code == 422  # removed stakeholder is not a valid enum


# --------------------------------------------------------------------------- #
# get
# --------------------------------------------------------------------------- #
def test_get_signal_found(client: TestClient) -> None:
    created = _create(client)
    resp = client.get(f"/signals/{created['id']}")
    assert resp.status_code == 200
    assert resp.json()["id"] == created["id"]


def test_get_signal_missing_returns_404(client: TestClient) -> None:
    resp = client.get(f"/signals/{uuid.uuid4()}")
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# analyze + get analysis
# --------------------------------------------------------------------------- #
def test_analyze_returns_analysis_and_run_meta(client: TestClient) -> None:
    created = _create(client)
    resp = client.post(f"/signals/{created['id']}/analyze")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["analysis"]["signal_id"] == created["id"]
    assert body["analysis"]["claims"]
    assert body["analysis"]["confidence"]["score"] == 0.8
    assert body["run"]["attempts_used"] == 1
    assert body["run"]["total_input_tokens"] == 120


def test_get_analysis_after_analyze(client: TestClient) -> None:
    created = _create(client)
    client.post(f"/signals/{created['id']}/analyze")
    resp = client.get(f"/signals/{created['id']}/analysis")
    assert resp.status_code == 200
    assert resp.json()["signal_id"] == created["id"]


def test_get_analysis_before_analyze_returns_404(client: TestClient) -> None:
    created = _create(client)
    resp = client.get(f"/signals/{created['id']}/analysis")
    assert resp.status_code == 404


def test_analyze_failure_returns_422_with_detail() -> None:
    failing = _build_client(grounded=False, max_attempts=2)
    created = _create(failing)
    resp = failing.post(f"/signals/{created['id']}/analyze")

    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["status"] == "failed_exhausted"
    assert detail["error_code"] == "semantic_validation_failed"
    assert detail["attempts_used"] == 2


def test_analyze_unconfigured_runner_returns_503() -> None:
    # No runner on app.state -> dependency raises 503.
    engine = make_engine()
    factory = make_session_factory(engine)

    def override_get_db():
        db = factory()
        try:
            yield db
        finally:
            db.close()

    app = create_app()
    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    resp = client.post("/signals", json={"source_type": "customer", "raw_text": "no runner"})
    # create needs the runner dependency too (service construction), so it 503s.
    assert resp.status_code == 503
