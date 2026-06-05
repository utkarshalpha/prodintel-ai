"""End-to-end API tests for the feature endpoints (Stage 1 -> Stage 2 over HTTP)."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from app.ai_runtime.retry_policy import RetryPolicy
from app.api.app import create_app
from app.api.deps import get_db
from app.stages.stage1.runner import build_stage1_runner
from app.stages.stage2.runner import build_stage2_runner

from tests.app_helpers import Stage1FakeClient, Stage2FakeClient, make_engine, make_session_factory


def _build_client(*, stage2_mode: str = "cluster_all", max_attempts: int = 3) -> TestClient:
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
    app.state.stage2_runner = build_stage2_runner(
        Stage2FakeClient(mode=stage2_mode), retry_policy=RetryPolicy(max_attempts=max_attempts)
    )
    return TestClient(app)


def _create_and_analyze(client: TestClient, text: str) -> str:
    created = client.post("/signals", json={"source_type": "customer", "raw_text": text})
    assert created.status_code == 201, created.text
    sid = created.json()["id"]
    analyzed = client.post(f"/signals/{sid}/analyze")
    assert analyzed.status_code == 200, analyzed.text
    return sid


@pytest.fixture
def client() -> TestClient:
    return _build_client()


def test_extract_features_endpoint(client: TestClient) -> None:
    s1 = _create_and_analyze(client, "Mobile checkout fails on pay")
    s2 = _create_and_analyze(client, "Customers cannot complete payment on phones")

    resp = client.post("/features/extract", json={"signal_ids": [s1, s2]})
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert len(body["features"]) == 1
    feature = body["features"][0]
    assert set(feature["source_signal_ids"]) == {s1, s2}
    assert feature["jtbd"].lower().startswith("when")
    assert body["run"]["attempts_used"] == 1


def test_get_and_list_features(client: TestClient) -> None:
    s1 = _create_and_analyze(client, "A signal to cluster")
    extract = client.post("/features/extract", json={"signal_ids": [s1]})
    feature_id = extract.json()["features"][0]["id"]

    got = client.get(f"/features/{feature_id}")
    assert got.status_code == 200
    assert got.json()["id"] == feature_id

    listed = client.get("/features", params={"signal_id": s1})
    assert listed.status_code == 200
    assert feature_id in {f["id"] for f in listed.json()}


def test_get_missing_feature_404(client: TestClient) -> None:
    assert client.get(f"/features/{uuid.uuid4()}").status_code == 404


def test_extract_unanalyzed_signal_422(client: TestClient) -> None:
    created = client.post("/signals", json={"source_type": "customer", "raw_text": "unanalyzed"})
    sid = created.json()["id"]
    resp = client.post("/features/extract", json={"signal_ids": [sid]})
    assert resp.status_code == 422
    assert sid in resp.json()["detail"]["signal_ids"]


def test_extract_failure_returns_422() -> None:
    failing = _build_client(stage2_mode="fabricate", max_attempts=2)
    s1 = _create_and_analyze(failing, "A signal that will fail extraction")
    resp = failing.post("/features/extract", json={"signal_ids": [s1]})
    assert resp.status_code == 422
    assert resp.json()["detail"]["error_code"] == "semantic_validation_failed"


def test_extract_without_runner_503() -> None:
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
    # stage1 present so we can create+analyze, but no stage2 runner.
    app.state.stage1_runner = build_stage1_runner(Stage1FakeClient(grounded=True))
    client = TestClient(app)

    resp = client.post("/features/extract", json={"signal_ids": [str(uuid.uuid4())]})
    assert resp.status_code == 503
