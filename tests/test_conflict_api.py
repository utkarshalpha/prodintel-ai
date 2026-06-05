"""End-to-end API tests for the conflict endpoints (Stage 1 -> 2 -> 3 over HTTP)."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from app.ai_runtime.retry_policy import RetryPolicy
from app.api.app import create_app
from app.api.deps import get_db
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


def _build_client(*, stage3_mode: str = "conflict", max_attempts: int = 3) -> TestClient:
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
    app.state.stage1_runner = build_stage1_runner(Stage1FakeClient(grounded=True))
    app.state.stage2_runner = build_stage2_runner(Stage2FakeClient(mode="cluster_all"))
    app.state.stage3_runner = build_stage3_runner(
        Stage3FakeClient(mode=stage3_mode), retry_policy=RetryPolicy(max_attempts=max_attempts)
    )
    return TestClient(app)


def _feature_over_two_stakeholders(client: TestClient) -> str:
    sales = client.post("/signals", json={"source_type": "sales", "raw_text": "Enterprise SSO is critical"}).json()
    eng = client.post("/signals", json={"source_type": "engineering", "raw_text": "SSO is high risk"}).json()
    client.post(f"/signals/{sales['id']}/analyze")
    client.post(f"/signals/{eng['id']}/analyze")
    extract = client.post("/features/extract", json={"signal_ids": [sales["id"], eng["id"]]})
    return extract.json()["features"][0]["id"]


@pytest.fixture
def client() -> TestClient:
    return _build_client()


def test_detect_conflicts_endpoint(client: TestClient) -> None:
    feature_id = _feature_over_two_stakeholders(client)

    resp = client.post("/conflicts/detect", json={"feature_ids": [feature_id]})
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert len(body["conflicts"]) == 1
    conflict = body["conflicts"][0]
    assert conflict["subject_id"] == feature_id
    assert set(conflict["stakeholders"]) == {"sales", "engineering"}
    assert conflict["evidence_signal_ids"]  # evidence-traceable
    assert {p["stance"] for p in conflict["positions"]} == {"advocate", "risk_flag"}
    assert body["run"]["attempts_used"] == 1


def test_get_and_list_conflicts(client: TestClient) -> None:
    feature_id = _feature_over_two_stakeholders(client)
    detect = client.post("/conflicts/detect", json={"feature_ids": [feature_id]})
    conflict_id = detect.json()["conflicts"][0]["id"]

    got = client.get(f"/conflicts/{conflict_id}")
    assert got.status_code == 200
    assert got.json()["id"] == conflict_id

    listed = client.get("/conflicts", params={"subject_id": feature_id})
    assert listed.status_code == 200
    assert conflict_id in {c["id"] for c in listed.json()}


def test_get_missing_conflict_404(client: TestClient) -> None:
    assert client.get(f"/conflicts/{uuid.uuid4()}").status_code == 404


def test_detect_missing_feature_404(client: TestClient) -> None:
    resp = client.post("/conflicts/detect", json={"feature_ids": [str(uuid.uuid4())]})
    assert resp.status_code == 404


def test_detect_failure_returns_422() -> None:
    failing = _build_client(stage3_mode="fabricate", max_attempts=2)
    feature_id = _feature_over_two_stakeholders(failing)
    resp = failing.post("/conflicts/detect", json={"feature_ids": [feature_id]})
    assert resp.status_code == 422
    assert resp.json()["detail"]["error_code"] == "semantic_validation_failed"


def test_detect_without_runner_503() -> None:
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
    client = TestClient(app)  # no stage3 runner configured
    resp = client.post("/conflicts/detect", json={"feature_ids": [str(uuid.uuid4())]})
    assert resp.status_code == 503
