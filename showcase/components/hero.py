"""Hero section: title, value proposition, key metrics, architecture summary, callout.

Presentation only -- all content is read from the snapshot or is static framing copy.
"""

from __future__ import annotations

import streamlit as st

from components.sections import render_metric_row

__all__ = ["render_hero"]

_ARCH_SUMMARY = (
    "Python · FastAPI · SQLAlchemy 2.0 · Pydantic v2 · Claude (tool-use) · "
    "ChromaDB (RAG) · Alembic · 12 ADRs · 493 tests"
)


def render_hero(snapshot: dict) -> None:
    """Render the top-of-page hero, key metrics, and the framing callout."""

    st.title("ProdIntel AI")
    st.markdown("#### Conflicting stakeholder signals → defensible, evidence-traceable product decisions")
    st.markdown(
        "> **Not a chatbot — a decision-support system.** Every recommendation traces back to the exact "
        "stakeholder quote that produced it, and any output the system cannot prove is *rejected, not shown*."
    )

    scenario = snapshot.get("meta", {}).get("scenario")
    if scenario:
        st.caption(f"📌 Scenario — {scenario}")

    render_metric_row(snapshot)
    st.caption(f"🧱 {_ARCH_SUMMARY}")

    st.info(
        "💡 **Why this isn't a RAG chatbot:** the model is *untrusted*. Every stage output passes "
        "deterministic validation gates and is rejected if it can't be substantiated — so a hallucinated "
        "decision can never be persisted or shown."
    )
