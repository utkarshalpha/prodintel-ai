"""End-to-end API tests for the decision endpoints (Stage 1 -> 2 -> 3 -> 4 over HTTP)."""

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
from app.stages.stage4.runner import build_stage4_runner

from tests.app_helpers import (
    Stage1FakeClient,
    Stage2FakeClient,
    Stage3FakeClient,
    Stage4FakeClient,
    make_engine,
    make_session_factory,
)


def _build_client(*, stage4_mode: str = "recommend", max_attempts: int = 3) -> TestClient:
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
    app.state.stage3_runner = build_stage3_runner(Stage3FakeClient(mode="conflict"))
    app.state.stage4_runner = build_stage4_runner(
        Stage4FakeClient(mode=stage4_mode), retry_policy=RetryPolicy(max_attempts=max_attempts)
    )
    return TestClient(app)


def _feature_with_conflict(client: TestClient) -> str:
    sales = client.post("/signals", json={"source_type": "sales", "raw_text": "Enterprise SSO is critical"}).json()
    eng = client.post("/signals", json={"source_type": "engineering", "raw_text": "SSO is high risk"}).json()
    client.post(f"/signals/{sales['id']}/analyze")
    client.post(f"/signals/{eng['id']}/analyze")
    extract = client.post("/features/extract", json={"signal_ids": [sales["id"], eng["id"]]})
    feature_id = extract.json()["features"][0]["id"]
    client.post("/conflicts/detect", json={"feature_ids": [feature_id]})
    return feature_id


@pytest.fixture
def client() -> TestClient:
    return _build_client()


def test_synthesize_decisions_endpoint(client: TestClient) -> None:
    feature_id = _feature_with_conflict(client)

    resp = client.post("/decisions/synthesize", json={"feature_ids": [feature_id]})
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert len(body["decisions"]) == 1
    decision = body["decisions"][0]
    assert decision["subject_id"] == feature_id
    assert decision["recommendation"] == "build_now"
    assert decision["evidence_signal_ids"]  # evidence-traceable
    assert decision["acknowledged_conflict_ids"]  # acknowledges the conflict over its subject
    assert decision["status"] == "proposed"
    assert body["run"]["attempts_used"] == 1


def test_get_and_list_decisions(client: TestClient) -> None:
    feature_id = _feature_with_conflict(client)
    synth = client.post("/decisions/synthesize", json={"feature_ids": [feature_id]})
    decision_id = synth.json()["decisions"][0]["id"]

    got = client.get(f"/decisions/{decision_id}")
    assert got.status_code == 200
    assert got.json()["id"] == decision_id

    listed = client.get("/decisions", params={"subject_id": feature_id})
    assert listed.status_code == 200
    assert decision_id in {d["id"] for d in listed.json()}


def test_get_missing_decision_404(client: TestClient) -> None:
    assert client.get(f"/decisions/{uuid.uuid4()}").status_code == 404


def test_synthesize_missing_feature_404(client: TestClient) -> None:
    resp = client.post("/decisions/synthesize", json={"feature_ids": [str(uuid.uuid4())]})
    assert resp.status_code == 404


def test_synthesize_failure_returns_422() -> None:
    failing = _build_client(stage4_mode="ignore_conflict", max_attempts=2)
    feature_id = _feature_with_conflict(failing)
    resp = failing.post("/decisions/synthesize", json={"feature_ids": [feature_id]})
    assert resp.status_code == 422
    assert resp.json()["detail"]["error_code"] == "semantic_validation_failed"


def test_synthesize_without_runner_503() -> None:
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
    client = TestClient(app)  # no stage4 runner configured
    resp = client.post("/decisions/synthesize", json={"feature_ids": [str(uuid.uuid4())]})
    assert resp.status_code == 503
