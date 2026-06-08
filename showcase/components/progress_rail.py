"""Pipeline rail: the six pipeline steps shown in the sidebar as a table of contents.

Lives in the sidebar so it stays visible while the main column scrolls. Pass ``active`` to
highlight the current step; with no active step every entry renders with a hollow marker.
"""

from __future__ import annotations

from collections.abc import Sequence

import streamlit as st

__all__ = ["render_progress_rail"]


def render_progress_rail(steps: Sequence[dict], active: str | None = None) -> None:
    """Render the pipeline steps as a vertical rail in the sidebar."""

    st.sidebar.markdown("### Pipeline")
    for step in steps:
        marker = "●" if step["key"] == active else "○"
        st.sidebar.markdown(f"{marker}&nbsp;&nbsp;{step['label']}", unsafe_allow_html=True)
