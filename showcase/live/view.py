"""Streamlit UI for Live Analysis mode.

Collects feedback (manual entry or file upload), runs it through L4 ingestion and the
Streamlit-free :mod:`showcase.live.runtime`, then renders the result with the **existing**
renderers (hero, progress rail, the six sections, footer). No rendering or snapshot logic
is defined here -- only input widgets, a diagnostics panel, and a partial-run status banner.

Import style: showcase-dir absolute (``components.*``, ``lib`` siblings, ``live.runtime``)
matching ``app.py``, since this module is loaded inside the ``streamlit run`` process where
the ``sys.path`` shim makes both the showcase dir and the repo root importable.
"""

from __future__ import annotations

import streamlit as st

from components.footer import render_footer
from components.hero import render_hero
from components.progress_rail import render_progress_rail
from components.sections import RENDERERS, STEPS
from live.runtime import run_local_analysis

from app.ai_contracts.enums import StakeholderType
from app.ingestion import IngestionError, format_for_filename, ingest, ingest_manual
from app.ingestion.contract import IngestionFormat

__all__ = ["render_live"]

_STAKEHOLDERS = [s.value for s in StakeholderType]

_EXAMPLE_ROWS = [
    {"stakeholder": "sales",
     "feedback": "We urgently need dark mode. Enterprise customers keep asking for it to close deals."},
    {"stakeholder": "engineering",
     "feedback": "Dark mode is risky and time-consuming to build across all screens."},
    {"stakeholder": "support",
     "feedback": "Customers report eye strain at night; dark mode would cut support tickets."},
]


# --------------------------------------------------------------------------- #
# State helpers
# --------------------------------------------------------------------------- #
def _clear_live_state() -> None:
    for key in ("live_result", "live_ingestion"):
        st.session_state.pop(key, None)


def _engine_banner() -> None:
    st.warning(
        "**Local heuristic engine** — deterministic, input-derived, and *illustrative*. It runs your "
        "feedback through the real pipeline (validation gates, provenance, `/why`) with **no API key**. "
        "It is **not** the Claude reasoning model and is not equivalent to Claude-quality analysis."
    )


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def _manual_rows() -> list[tuple[str | None, str]]:
    st.caption(
        "Enter stakeholder feedback (one row each). Use feedback from **different stakeholders** "
        "to surface conflicts. Add or remove rows with the editor controls."
    )
    seed = st.session_state.get("live_manual", [dict(stakeholder="", feedback="") for _ in range(3)])
    nonce = st.session_state.get("live_manual_nonce", 0)
    edited = st.data_editor(
        seed,
        num_rows="dynamic",
        use_container_width=True,
        key=f"live_editor_{nonce}",
        column_config={
            "stakeholder": st.column_config.SelectboxColumn(
                "Stakeholder", options=[""] + _STAKEHOLDERS, width="small"),
            "feedback": st.column_config.TextColumn("Feedback", width="large"),
        },
    )
    if hasattr(edited, "to_dict"):  # defensive: a DataFrame -> list[dict]
        edited = edited.to_dict("records")

    rows: list[tuple[str | None, str]] = []
    for row in edited:
        text = (row.get("feedback") or "").strip()
        if not text:
            continue
        stake = (row.get("stakeholder") or "").strip() or None
        rows.append((stake, text))
    return rows


def _upload_options(fmt: IngestionFormat) -> dict:
    options: dict = {}
    with st.expander("Advanced options"):
        if fmt is IngestionFormat.CSV:
            options["text_column"] = st.text_input("CSV text column", value="feedback")
            options["stakeholder_column"] = st.text_input("CSV stakeholder column", value="stakeholder")
        elif fmt in (IngestionFormat.TXT, IngestionFormat.PDF):
            options["granularity"] = st.selectbox("Segment by", ["paragraph", "line", "document"], index=0)
        else:
            st.caption("No options for this format.")
    if fmt is not IngestionFormat.CSV:
        st.caption(
            "ℹ Documents carry no stakeholder, so every signal uses the default above — "
            "upload a CSV with a stakeholder column (or use manual entry) to surface conflicts."
        )
    return options


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def _run(source: tuple[str, object], default_stakeholder: str) -> None:
    kind, payload = source
    stakeholder = StakeholderType(default_stakeholder)
    try:
        if kind == "manual":
            ingestion = ingest_manual(payload, default_stakeholder=stakeholder)  # type: ignore[arg-type]
        else:
            upload, options = payload  # type: ignore[misc]
            ingestion = ingest(upload.getvalue(), filename=upload.name,
                               default_stakeholder=stakeholder, **options)
    except IngestionError as exc:
        _clear_live_state()
        st.error(f"Could not read the input: {exc}")
        return

    st.session_state["live_ingestion"] = ingestion
    if ingestion.accepted_count == 0:
        st.session_state.pop("live_result", None)
        return
    with st.spinner("Running local analysis…"):
        try:
            st.session_state["live_result"] = run_local_analysis(list(ingestion.entries))
        except Exception as exc:  # surface, never white-screen (local demo, no secrets)
            st.session_state.pop("live_result", None)
            st.error(f"Analysis failed unexpectedly: {exc}")


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def _render_ingestion() -> None:
    ingestion = st.session_state.get("live_ingestion")
    if ingestion is None:
        return
    st.markdown("### Ingestion")
    cols = st.columns(3)
    cols[0].metric("Accepted", ingestion.accepted_count)
    cols[1].metric("Rejected", ingestion.rejected_count)
    cols[2].metric("Warnings", len(ingestion.warnings))

    if ingestion.accepted_count == 0:
        st.warning("No valid feedback to analyze — fix the rejected rows below and run again.")
    if ingestion.rejected:
        with st.expander(f"Rejected rows ({ingestion.rejected_count})"):
            for row in ingestion.rejected:
                detail = f" — {row.detail}" if row.detail else ""
                excerpt = f"  ·  _{row.excerpt}_" if row.excerpt else ""
                st.markdown(f"- `{row.locator}` — **{row.reason.value}**{detail}{excerpt}")
    if ingestion.warnings:
        with st.expander(f"Warnings ({len(ingestion.warnings)})"):
            for warning in ingestion.warnings:
                detail = f" — {warning.detail}" if warning.detail else ""
                st.markdown(f"- `{warning.locator}` — {warning.kind.value}{detail}")


def _render_results() -> None:
    result = st.session_state.get("live_result")
    if result is None:
        return
    snapshot = result.snapshot

    st.markdown("### Results")
    if result.succeeded:
        st.success("✅ Analysis complete — every stage passed the validation gates.")
    else:
        st.warning(
            f"⚠ Stage **{result.failed_stage}** did not complete — showing partial results "
            "(the pipeline degrades gracefully rather than failing the whole run)."
        )
        for note in result.stage_notes:
            if note.status in ("failed", "skipped") and note.detail:
                st.caption(f"• {note.stage}: {note.detail}")

    render_hero(snapshot)
    render_progress_rail(STEPS)
    st.markdown("## Generated analysis")
    for step in STEPS:
        st.subheader(step["label"])
        st.caption(step["blurb"])
        RENDERERS[step["key"]](snapshot)
        st.divider()
    render_footer(snapshot)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def render_live() -> None:
    st.markdown("## Live analysis")
    _engine_banner()

    method = st.radio("Input", ["Manual entry", "Upload file"], horizontal=True, key="live_method")
    default_stakeholder = st.selectbox(
        "Default stakeholder (documents & blank rows)", _STAKEHOLDERS, index=0, key="live_default_stake")

    if method == "Manual entry":
        if st.button("Load example"):
            st.session_state["live_manual"] = [dict(row) for row in _EXAMPLE_ROWS]
            st.session_state["live_manual_nonce"] = st.session_state.get("live_manual_nonce", 0) + 1
            _clear_live_state()
            st.rerun()
        rows = _manual_rows()
        source: tuple[str, object] = ("manual", rows)
        ready = bool(rows)
    else:
        upload = st.file_uploader("Upload feedback", type=["csv", "txt", "pdf", "docx"], key="live_upload")
        options: dict = {}
        if upload is not None:
            try:
                options = _upload_options(format_for_filename(upload.name))
            except IngestionError as exc:
                st.error(str(exc))
        source = ("upload", (upload, options))
        ready = upload is not None

    left, right = st.columns([1, 1])
    if left.button("Run analysis", type="primary", disabled=not ready):
        _run(source, default_stakeholder)
    if right.button("Start over"):
        _clear_live_state()
        st.rerun()

    _render_ingestion()
    _render_results()
