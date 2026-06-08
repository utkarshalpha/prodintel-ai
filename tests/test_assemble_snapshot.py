"""Phase L2 -- shared snapshot assembler tests.

The primary gate is the byte-identical regression: the refactored generator (now driving
PipelineService + assemble_snapshot) must reproduce the committed demo snapshot exactly, so
the Streamlit renderers need zero changes. Plus: assemble_snapshot from a live-shaped run,
graceful partial runs, and the presentation-only contract.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path

import pytest

# Initialize the api package first (the generator + explanation service import app.api).
import app.api.app  # noqa: F401,E402

from app.ai_contracts.enums import StakeholderType
from app.repositories.conflict_repository import ConflictRepository
from app.repositories.decision_repository import DecisionRepository
from app.repositories.feature_repository import FeatureRepository
from app.repositories.signal_repository import SignalRepository
from app.services.decision_explanation_service import DecisionExplanationService
from app.services.pipeline_service import FeedbackEntry, PipelineService
from app.ai_runtime.retry_policy import RetryPolicy
from app.stages.stage1.runner import build_stage1_runner
from app.stages.stage2.runner import build_stage2_runner
from app.stages.stage3.runner import build_stage3_runner
from app.stages.stage4.runner import build_stage4_runner
from showcase.lib.assemble import assemble_snapshot

from tests.app_helpers import (
    Stage1FakeClient,
    Stage2FakeClient,
    Stage3FakeClient,
    Stage4FakeClient,
    make_engine,
    make_session_factory,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SNAPSHOT_PATH = _REPO_ROOT / "showcase" / "data" / "demo_snapshot.json"
_GENERATOR_PATH = _REPO_ROOT / "scripts" / "build_demo_snapshot.py"


def _load_generator():
    """Import scripts/build_demo_snapshot.py as a module (it is not a package)."""

    spec = importlib.util.spec_from_file_location("build_demo_snapshot", _GENERATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # __name__ != "__main__" -> main() does NOT run (no file write)
    return module


@pytest.fixture
def session():
    db = make_session_factory(make_engine())()
    try:
        yield db
    finally:
        db.close()


def _entries():
    return [
        FeedbackEntry(StakeholderType.SALES, "Enterprise SSO is critical to close the Acme deal."),
        FeedbackEntry(StakeholderType.ENGINEERING, "SSO is high risk and needs six to eight weeks of work."),
    ]


def _service(session, *, s2="cluster_all", max_attempts=2):
    rp = RetryPolicy(max_attempts=max_attempts)
    explanation = DecisionExplanationService(
        DecisionRepository(session), FeatureRepository(session),
        ConflictRepository(session), SignalRepository(session),
    )
    return PipelineService(
        session,
        stage1_runner=build_stage1_runner(Stage1FakeClient(grounded=True), retry_policy=rp),
        stage2_runner=build_stage2_runner(Stage2FakeClient(mode=s2), retry_policy=rp),
        stage3_runner=build_stage3_runner(Stage3FakeClient(mode="conflict"), retry_policy=rp),
        stage4_runner=build_stage4_runner(Stage4FakeClient(mode="recommend"), retry_policy=rp),
        explanation_service=explanation,
    )


# --------------------------------------------------------------------------- #
# Primary gate: the committed demo snapshot is reproduced byte-for-byte.
# --------------------------------------------------------------------------- #
def test_demo_snapshot_is_byte_identical() -> None:
    gen = _load_generator()
    produced = gen.build_snapshot()  # in-memory; does not write the artifact

    committed_text = _SNAPSHOT_PATH.read_text(encoding="utf-8")
    committed = json.loads(committed_text)
    assert produced == committed, "assemble_snapshot refactor changed the demo snapshot values"

    # Byte-identical to what main() writes: json.dumps(..., indent=2) + trailing newline.
    produced_text = json.dumps(produced, ensure_ascii=False, indent=2) + "\n"
    assert produced_text == committed_text, "assemble_snapshot refactor changed snapshot serialization/order"


# --------------------------------------------------------------------------- #
# assemble_snapshot from a live-shaped PipelineRunResult
# --------------------------------------------------------------------------- #
def test_assemble_from_pipeline_run(session) -> None:
    run = _service(session).analyze(_entries())
    snap = assemble_snapshot(run, scenario="Live analysis", generated_by="test")

    assert set(snap.keys()) == {
        "meta", "signals", "analyzed_signals", "features", "unassigned_signal_ids",
        "conflicts", "decisions", "framework_pool", "framework_citations", "why",
    }
    assert "invalid_example" not in snap                       # generator-owned, not here
    assert snap["meta"] == {"scenario": "Live analysis", "schema_version": "1.0",
                            "generated_by": "test", "note": None}
    assert len(snap["signals"]) == 2 and len(snap["analyzed_signals"]) == 2
    assert len(snap["features"]) >= 1 and len(snap["decisions"]) >= 1
    assert snap["why"] is not None
    assert all("quoted_text" in c for a in snap["analyzed_signals"] for c in a["claims"])


def test_assemble_partial_run_degrades_gracefully(session) -> None:
    # Stage 2 fails -> no features/conflicts/decisions/explanation.
    run = _service(session, s2="fabricate", max_attempts=1).analyze(_entries())
    snap = assemble_snapshot(run, scenario="Partial")

    assert len(snap["signals"]) == 2                           # Stage 1 succeeded
    assert snap["features"] == [] and snap["decisions"] == []
    assert snap["framework_pool"] == [] and snap["framework_citations"] == {}
    assert snap["why"] is None


# --------------------------------------------------------------------------- #
# Contract: presentation-only (no session, no LLM, no PipelineService)
# --------------------------------------------------------------------------- #
def test_assembler_is_presentation_only() -> None:
    params = inspect.signature(assemble_snapshot).parameters
    assert "session" not in params               # reads off the result's live entities, no session param
    assert "run_result" in params                # the canonical input
    import showcase.lib.assemble as mod
    assert "json" in dir(mod)                    # pure transform (stdlib only at runtime)
