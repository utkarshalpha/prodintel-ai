"""Shared recruiter-facing presentation constants (single source of truth).

Values that would otherwise drift across components -- the test count and the canonical
repository URL -- live here so the hero and footer can never disagree.
"""

from __future__ import annotations

__all__ = ["TEST_COUNT", "REPO_URL"]

# Total tests in the suite: 565 passing + 4 environment-gated chroma-integration skips.
TEST_COUNT = 569

# The project repository (NOT the bare profile) -- used for the footer GitHub/README links.
REPO_URL = "https://github.com/utkarshalpha/prodintel-ai"
