"""Cached loader for the demo snapshot -- the single source of truth for the showcase.

``read_snapshot`` is a pure, Streamlit-free function (unit-testable anywhere).
``load_snapshot`` is the app-facing entry point: it caches the read with
``st.cache_data`` when Streamlit is available, and falls back to ``lru_cache`` so the
module still imports cleanly in a plain test environment.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

__all__ = ["SNAPSHOT_PATH", "read_snapshot", "load_snapshot"]

# showcase/lib/snapshot.py -> showcase/data/demo_snapshot.json
SNAPSHOT_PATH = Path(__file__).resolve().parents[1] / "data" / "demo_snapshot.json"


def read_snapshot(path: Path | None = None) -> dict:
    """Read and parse the demo snapshot JSON (no Streamlit dependency)."""

    target = path or SNAPSHOT_PATH
    return json.loads(target.read_text(encoding="utf-8"))


try:  # pragma: no cover - exercised only with Streamlit installed
    import streamlit as st

    @st.cache_data(show_spinner=False)
    def load_snapshot() -> dict:
        """Return the demo snapshot, cached for the session."""

        return read_snapshot()

except ModuleNotFoundError:  # pragma: no cover - test/CI without Streamlit

    @lru_cache(maxsize=1)
    def load_snapshot() -> dict:
        """Return the demo snapshot, cached (lru fallback when Streamlit is absent)."""

        return read_snapshot()
