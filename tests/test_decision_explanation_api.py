"""Phase 5 (D) -- API tests for GET /decisions/{id}/why.

200 success with full provenance shape, 404 for a missing decision, and deterministic
repeated responses. Reuses the Stage 1→4 client builder from the decision API tests.
"""

from __future__ import annotations

from uuid import uuid4

from tests.test_decision_api import _build_client, _feature_with_conflict


def _synthesize_decision(client) -> str:
    feature_id = _feature_with_conflict(client)
    synth = client.post("/decisions/synthesize", json={"feature_ids": [feature_id]})
    return synth.json()["decisions"][0]["id"], feature_id


def test_why_endpoint_returns_full_explanation() -> None:
    client = _build_client()
    decision_id, feature_id = _synthesize_decision(client)

    resp = client.get(f"/decisions/{decision_id}/why")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["decision"]["id"] == decision_id
    assert body["subject_feature"]["id"] == feature_id
    assert body["direct_evidence"]
    assert body["conflicts"] and body["conflicts"][0]["parties"]
    assert body["signals"] and all(s["raw_text"] for s in body["signals"])
    assert body["integrity"]["complete"] is True
    assert body["integrity"]["unresolved_signal_ids"] == []
    assert body["meta"]["decision_id"] == decision_id
    assert body["schema_version"] == "1.0"


def test_why_endpoint_404_for_missing_decision() -> None:
    client = _build_client()
    assert client.get(f"/decisions/{uuid4()}/why").status_code == 404


def test_why_endpoint_is_deterministic() -> None:
    client = _build_client()
    decision_id, _ = _synthesize_decision(client)

    first = client.get(f"/decisions/{decision_id}/why").json()
    second = client.get(f"/decisions/{decision_id}/why").json()
    assert first == second
