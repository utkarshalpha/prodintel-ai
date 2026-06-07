"""ProdIntel AI -- recruiter-facing Streamlit showcase.

Single-page narrative driven entirely by ``showcase/data/demo_snapshot.json`` (no live
API). Showcase mode walks the six pipeline steps; Architecture mode renders the project's
diagrams. Hero, progress rail, sections, and footer are composed from ``components/``.

Run:  streamlit run showcase/app.py
"""

from __future__ import annotations

import streamlit as st

from components.architecture import render_architecture
from components.footer import render_footer
from components.hero import render_hero
from components.progress_rail import render_progress_rail
from components.sections import (
    render_analysis,
    render_conflicts,
    render_decision,
    render_features,
    render_signals,
    render_why,
)
from lib.snapshot import SNAPSHOT_PATH, load_snapshot

# The six narrative steps (Showcase mode). Each becomes an empty container for now.
STEPS = [
    {"key": "signals", "label": "① Signals", "blurb": "The conflicting stakeholder inputs"},
    {"key": "analysis", "label": "② Analysis", "blurb": "Claims anchored to the source text"},
    {"key": "features", "label": "③ Features", "blurb": "Signals clustered into product features"},
    {"key": "conflicts", "label": "④ Conflicts", "blurb": "Genuine stakeholder disagreement, surfaced"},
    {"key": "decision", "label": "⑤ Decision", "blurb": "A ranked, evidence-backed recommendation"},
    {"key": "why", "label": "⑥ Why", "blurb": "Full provenance back to the original quote"},
]

# Dispatch: step key -> the renderer that fills its section from the snapshot.
RENDERERS = {
    "signals": render_signals,
    "analysis": render_analysis,
    "features": render_features,
    "conflicts": render_conflicts,
    "decision": render_decision,
    "why": render_why,
}

def _inject_base_style() -> None:
    """Light, defensive presentation tweaks (no behavior change)."""

    st.markdown(
        "<style>.block-container{padding-top:2.2rem;max-width:1100px;}</style>",
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="ProdIntel AI — Decision Intelligence", page_icon="🧭", layout="wide")
    _inject_base_style()

    try:
        snapshot = load_snapshot()
    except FileNotFoundError:
        st.error(
            f"Demo snapshot not found at `{SNAPSHOT_PATH}`.\n\n"
            "Generate it with: `python scripts/build_demo_snapshot.py`"
        )
        st.stop()

    mode = st.sidebar.radio("Mode", ["Showcase", "Architecture"], index=0)

    render_hero(snapshot)

    if mode == "Showcase":
        render_progress_rail(STEPS)
        st.markdown("## Guided walkthrough")
        for step in STEPS:
            st.subheader(step["label"])
            st.caption(step["blurb"])
            RENDERERS[step["key"]](snapshot)
            st.divider()
    else:
        st.markdown("## Architecture")
        render_architecture()

    render_footer(snapshot)


main()
