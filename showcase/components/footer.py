"""Footer: the exit links and a one-line credibility strip.

Scaffold only -- the demo-video and resume links are placeholders until those assets exist.
"""

from __future__ import annotations

import streamlit as st

__all__ = ["render_footer"]

_REPO_URL = "https://github.com/utkarshalpha"


def render_footer(snapshot: dict) -> None:
    """Render the bottom-of-page links and credibility line."""

    st.divider()
    cols = st.columns(4)
    cols[0].markdown(f"[GitHub]({_REPO_URL})")
    cols[1].markdown(f"[README]({_REPO_URL}#readme)")
    cols[2].markdown("Demo video — _coming soon_")
    cols[3].markdown("Resume — _coming soon_")
    st.caption("493 tests passing · RAG-grounded decisions · full decision provenance")
