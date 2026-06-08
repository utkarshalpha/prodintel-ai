"""ProdIntel AI -- recruiter-facing Streamlit app.

Three modes, selected in the sidebar:

* **Showcase** -- the deterministic recruiter walkthrough, driven entirely by
  ``showcase/data/demo_snapshot.json`` (no live pipeline, no API key). Unchanged.
* **Live Analysis** -- runs the *real* pipeline on user-supplied feedback via the local
  heuristic engine (no API key), reusing the same renderers as Showcase.
* **Architecture** -- renders the project's source-of-truth diagrams. Unchanged.

Run:  streamlit run showcase/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# --- path shim --------------------------------------------------------------------------
# `streamlit run` puts the script's own directory (showcase/) on sys.path[0]. That dir
# contains THIS file, app.py -- which would shadow the backend `app/` package, so a Live-mode
# `import app.api.app` would re-import this script (circular import). Fix: put the repo root
# AHEAD of the showcase dir, so `import app` resolves to the backend package; keep the
# showcase dir present (after it) for the `components`/`lib`/`live` imports below.
_HERE = Path(__file__).resolve().parent          # .../showcase
_ROOT = _HERE.parent                              # repo root
if str(_HERE) not in sys.path:
    sys.path.append(str(_HERE))
if str(_ROOT) in sys.path:
    sys.path.remove(str(_ROOT))
sys.path.insert(0, str(_ROOT))                    # repo root first -> backend `app` wins over app.py

import streamlit as st

from components.architecture import render_architecture
from components.footer import render_footer
from components.hero import render_hero
from components.progress_rail import render_progress_rail
from components.sections import RENDERERS, STEPS
from lib.snapshot import SNAPSHOT_PATH, load_snapshot
from live.view import render_live


def _inject_base_style() -> None:
    """Light, defensive presentation tweaks (no behavior change)."""

    st.markdown(
        "<style>.block-container{padding-top:2.2rem;max-width:1100px;}</style>",
        unsafe_allow_html=True,
    )


def _demo_snapshot() -> dict:
    """Load the committed demo snapshot (Showcase/Architecture only)."""

    try:
        return load_snapshot()
    except FileNotFoundError:
        st.error(
            f"Demo snapshot not found at `{SNAPSHOT_PATH}`.\n\n"
            "Generate it with: `python scripts/build_demo_snapshot.py`"
        )
        st.stop()


def main() -> None:
    st.set_page_config(page_title="ProdIntel AI — Decision Intelligence", page_icon="🧭", layout="wide")
    _inject_base_style()

    mode = st.sidebar.radio("Mode", ["Showcase", "Live Analysis", "Architecture"], index=0)

    # Live Analysis is self-contained (no demo snapshot dependency).
    if mode == "Live Analysis":
        render_live()
        return

    snapshot = _demo_snapshot()
    render_hero(snapshot)

    if mode == "Showcase":
        render_progress_rail(STEPS)
        st.markdown("## Guided walkthrough")
        for step in STEPS:
            st.subheader(step["label"])
            st.caption(step["blurb"])
            RENDERERS[step["key"]](snapshot)
            st.divider()
    else:  # Architecture
        st.markdown("## Architecture")
        render_architecture()

    render_footer(snapshot)


main()
