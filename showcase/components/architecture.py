"""Architecture mode: render the approved diagrams as cards.

Each diagram is rendered with mermaid.js (via the built-in components.html), with the
raw Mermaid source embedded in a "view source" expander as a graceful fallback when the
renderer is unavailable.
"""

from __future__ import annotations

import streamlit as st

from lib.diagrams import DIAGRAMS

__all__ = ["render_architecture", "render_mermaid"]


def render_mermaid(code: str, height: int = 480) -> None:
    """Render a Mermaid diagram; fall back to showing the source if JS is unavailable."""

    try:
        import streamlit.components.v1 as components

        html = (
            '<div class="mermaid">' + code + "</div>"
            "<script type=\"module\">"
            "import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';"
            "mermaid.initialize({ startOnLoad: true, theme: 'neutral' });"
            "</script>"
        )
        components.html(html, height=height, scrolling=True)
    except Exception:  # pragma: no cover - headless / no components runtime
        st.code(code, language="text")


def render_architecture() -> None:
    """Render every approved diagram as a card (rendered diagram + source expander)."""

    if not DIAGRAMS:
        st.info("No architecture diagrams are available.")
        return
    st.caption("The system's design, end to end. Rendered from the project's source-of-truth diagrams.")
    for diagram in DIAGRAMS:
        with st.container(border=True):
            st.subheader(diagram["title"])
            st.caption(diagram["description"])
            render_mermaid(diagram["mermaid"], diagram.get("height", 480))
            with st.expander("View Mermaid source"):
                st.code(diagram["mermaid"], language="text")
