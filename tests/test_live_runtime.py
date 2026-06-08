"""Phase L5 -- Live Analysis runtime + reuse/non-regression guards.

The substantive logic lives in the Streamlit-free ``showcase.live.runtime``; it is tested
directly here. The Streamlit UI (``view.py``) is exercised only via a gated AppTest smoke
test (skipped when Streamlit is absent).
"""

from __future__ import annotations

import json

import pytest

# Importing the runtime initialises the api package (explanation-service import cycle).
from showcase.live.runtime import LIVE_SCENARIO, LiveRunResult, run_local_analysis

from app.ai_contracts.enums import StakeholderType as S
from app.ingestion import FeedbackEntry, ingest_manual

_CANONICAL_KEYS = {
    "meta", "signals", "analyzed_signals", "features", "unassigned_signal_ids",
    "conflicts", "decisions", "framework_pool", "framework_citations", "why",
}

_OPPOSING = [
    FeedbackEntry(S.SALES, "We urgently need dark mode. Enterprise customers keep asking for it."),
    FeedbackEntry(S.ENGINEERING, "Dark mode is risky and time-consuming to build across all screens."),
]


# --------------------------------------------------------------------------- #
# Runtime: end-to-end success
# --------------------------------------------------------------------------- #
def test_run_local_analysis_succeeds() -> None:
    result = run_local_analysis(_OPPOSING)
    assert isinstance(result, LiveRunResult)
    assert result.succeeded and result.failed_stage is None
    snap = result.snapshot
    assert set(snap.keys()) == _CANONICAL_KEYS                 # built by assemble_snapshot
    assert snap["meta"]["scenario"] == LIVE_SCENARIO
    assert result.counts == {"signals": 2, "features": 1, "conflicts": 1, "decisions": 1}
    assert snap["why"] is not None
    assert "invalid_example" not in snap                       # generator-only, never here


def test_outputs_are_input_derived() -> None:
    dark = run_local_analysis(_OPPOSING).snapshot
    other = run_local_analysis([
        FeedbackEntry(S.SUPPORT, "Customers want faster export of reports to csv."),
        FeedbackEntry(S.SALES, "Report export speed is a deal driver."),
    ]).snapshot
    dark_titles = " ".join(f["title"].lower() for f in dark["features"])
    other_titles = " ".join(f["title"].lower() for f in other["features"])
    assert "dark" in dark_titles or "mode" in dark_titles
    assert other_titles != dark_titles


def test_determinism_same_input_same_content() -> None:
    a = run_local_analysis(_OPPOSING).snapshot
    b = run_local_analysis(_OPPOSING).snapshot
    # ids differ (fresh in-memory DBs) but derived content is identical.
    assert [f["title"] for f in a["features"]] == [f["title"] for f in b["features"]]
    assert [d["recommendation"] for d in a["decisions"]] == [d["recommendation"] for d in b["decisions"]]
    assert ([[c["text"] for c in s["claims"]] for s in a["analyzed_signals"]]
            == [[c["text"] for c in s["claims"]] for s in b["analyzed_signals"]])


def test_empty_entries_raises() -> None:
    with pytest.raises(ValueError):
        run_local_analysis([])


# --------------------------------------------------------------------------- #
# Integration: ingestion -> pipeline -> snapshot
# --------------------------------------------------------------------------- #
def test_ingestion_to_runtime_integration() -> None:
    ingestion = ingest_manual([
        ("sales", "We urgently need dark mode for our enterprise customers."),
        ("engineering", "Dark mode is risky and time-consuming to build across all screens."),
    ])
    assert ingestion.accepted_count == 2
    result = run_local_analysis(list(ingestion.entries))
    assert result.succeeded
    assert result.counts["signals"] == 2 and result.counts["decisions"] >= 1


# --------------------------------------------------------------------------- #
# The UI must never hold ORM entities: the result is fully JSON-serializable.
# --------------------------------------------------------------------------- #
def test_result_is_detached_and_serializable() -> None:
    result = run_local_analysis(_OPPOSING)
    json.dumps(result.snapshot)                                # would raise on any ORM object
    assert isinstance(result.snapshot, dict)
    assert all(isinstance(note.stage, str) for note in result.stage_notes)


# --------------------------------------------------------------------------- #
# Partial-run: a fatal Stage 1 is reported, not raised; snapshot still renders.
# --------------------------------------------------------------------------- #
def test_partial_run_surfaces_failed_stage() -> None:
    result = run_local_analysis([FeedbackEntry(S.SALES, "!!!"), FeedbackEntry(S.SUPPORT, "...")])
    assert not result.succeeded
    assert result.failed_stage == "signal_analysis"
    assert result.counts["signals"] == 0
    assert set(result.snapshot.keys()) == _CANONICAL_KEYS      # degrades, still renderable
    json.dumps(result.snapshot)


# --------------------------------------------------------------------------- #
# Reuse guard: the six-step walkthrough is a single shared source of truth.
# --------------------------------------------------------------------------- #
def test_walkthrough_is_shared_source_of_truth() -> None:
    from showcase.components.sections import (
        RENDERERS,
        STEPS,
        render_analysis,
        render_signals,
    )

    assert [s["key"] for s in STEPS] == ["signals", "analysis", "features", "conflicts", "decision", "why"]
    assert set(RENDERERS) == {s["key"] for s in STEPS}         # dispatch matches the steps
    assert RENDERERS["signals"] is render_signals              # the real renderers, not copies
    assert RENDERERS["analysis"] is render_analysis


def test_runtime_uses_assemble_snapshot() -> None:
    # The runtime constructs snapshots only via assemble_snapshot (no bespoke builder).
    import inspect

    import showcase.live.runtime as runtime
    source = inspect.getsource(runtime)
    assert "assemble_snapshot(" in source
    assert "from showcase.lib.assemble import assemble_snapshot" in source


# --------------------------------------------------------------------------- #
# Showcase non-regression: the committed demo snapshot still matches byte-for-byte.
# (The authoritative check lives in test_assemble_snapshot.py; this is a fast guard
# that the STEPS/RENDERERS relocation didn't disturb the demo artifact.)
# --------------------------------------------------------------------------- #
def test_demo_snapshot_keys_unchanged() -> None:
    from showcase.lib.snapshot import read_snapshot

    snap = read_snapshot()
    assert set(snap.keys()) >= _CANONICAL_KEYS                 # demo includes the canonical set


# --------------------------------------------------------------------------- #
# Gated Streamlit UI smoke test (skipped when Streamlit is not installed).
# --------------------------------------------------------------------------- #
def _button(app_test, label):
    return next(b for b in app_test.button if b.label == label)


def test_app_runs_under_apptest() -> None:
    pytest.importorskip("streamlit")
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parents[1] / "showcase" / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=60).run()
    assert not at.exception                                     # default Showcase mode renders cleanly
    assert list(at.sidebar.radio[0].options) == ["Showcase", "Live Analysis", "Architecture"]

    # Switch to Live mode -> input form (banner + data_editor + selectbox) renders.
    at.sidebar.radio[0].set_value("Live Analysis").run()
    assert not at.exception

    # Drive the full live flow through the UI: example -> run -> ingest -> pipeline -> render.
    _button(at, "Load example").click().run()
    assert not at.exception
    _button(at, "Run analysis").click().run()
    assert not at.exception
    assert any("Analysis complete" in s.value for s in at.success)  # results rendered via shared renderers
