"""Footer: recruiter-facing exit links and a one-line credibility strip."""

from __future__ import annotations

import streamlit as st

from lib.meta import REPO_URL, TEST_COUNT

__all__ = ["render_footer"]

_LINKEDIN_URL = "https://www.linkedin.com/in/utkaxh/"
_EMAIL = "utkarsh7854@gmail.com"


def render_footer(snapshot: dict) -> None:
    """Render the bottom-of-page links and credibility line."""

    st.divider()
    cols = st.columns(4)
    cols[0].markdown(f"[GitHub repo]({REPO_URL})")
    cols[1].markdown(f"[README]({REPO_URL}#readme)")
    cols[2].markdown(f"[LinkedIn]({_LINKEDIN_URL})")
    cols[3].markdown(f"[Email](mailto:{_EMAIL})")
    st.caption(f"{TEST_COUNT} tests · RAG-grounded decisions · full decision provenance")
