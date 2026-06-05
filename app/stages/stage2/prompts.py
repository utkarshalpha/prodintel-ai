"""Prompt package and tool identity for Stage 2 (Feature Extraction)."""

from __future__ import annotations

from typing import Sequence
from uuid import UUID

from pydantic import BaseModel

__all__ = [
    "STAGE2_PROMPT_VERSION",
    "STAGE2_TOOL_NAME",
    "STAGE2_TOOL_DESCRIPTION",
    "SignalForPrompt",
    "build_system_prompt",
    "build_user_prompt",
]

STAGE2_PROMPT_VERSION = "stage2-v1"
STAGE2_TOOL_NAME = "emit_features"

STAGE2_TOOL_DESCRIPTION = (
    "Cluster the provided analyzed signals into normalized product features. Merge "
    "duplicate or near-duplicate requests into one feature. Every feature MUST cite "
    "the signal_ids it derives from (a subset of the inputs only). Every input "
    "signal must appear in exactly one feature's source_signal_ids OR in "
    "unassigned_signal_ids -- never both, never neither."
)


class SignalForPrompt(BaseModel):
    """Minimal view of one analyzed signal, used to render the prompt."""

    signal_id: UUID
    stakeholder_type: str
    urgency: int
    intent: str
    claims: list[str]


def build_system_prompt() -> str:
    """Static instructions for the feature-extraction stage."""

    return (
        "You are the Feature Extraction stage of an enterprise product-decision "
        "system. You convert a set of analyzed stakeholder signals into a smaller "
        "set of normalized product features. Respond ONLY by calling the "
        f"{STAGE2_TOOL_NAME!r} tool; never reply with free text.\n\n"
        "Rules:\n"
        "1. Merge duplicate or overlapping requests across signals into a single "
        "feature -- do not emit one feature per signal.\n"
        "2. Normalize each feature's title into a concise, product-style name "
        "(<= 120 chars); avoid stakeholder-specific phrasing.\n"
        "3. Preserve provenance: list in source_signal_ids every signal_id that the "
        "feature was derived from. Use only the signal_ids provided.\n"
        "4. Write a Jobs-to-be-Done statement for each feature in the form "
        "'When <situation>, I want <motivation>, so I can <outcome>'.\n"
        "5. Produce a confidence object per feature: a score in [0, 1] and a "
        "components map (e.g. 'cluster_cohesion', 'cross_stakeholder_support').\n"
        "6. Account for EVERY input signal: each signal_id must appear in exactly "
        "one feature's source_signal_ids or in unassigned_signal_ids.\n"
        "A downstream validator re-checks that every cited signal_id is real and "
        "that the inputs are fully partitioned, and rejects the output otherwise."
    )


def build_user_prompt(signals: Sequence[SignalForPrompt]) -> str:
    """Render the analyzed signals for clustering.

    Each signal is printed with an explicit ``signal_id:`` line so the model (and
    the provenance validator) share an unambiguous identifier.
    """

    blocks: list[str] = []
    for position, signal in enumerate(signals, start=1):
        claim_lines = "\n".join(f"      - {claim}" for claim in signal.claims) or "      - (none)"
        blocks.append(
            f"[{position}] signal_id: {signal.signal_id} | source: {signal.stakeholder_type} "
            f"| urgency: {signal.urgency}\n"
            f"    intent: {signal.intent}\n"
            f"    claims:\n{claim_lines}"
        )
    body = "\n".join(blocks)
    return (
        "Cluster the following analyzed signals into normalized product features. "
        "Account for every signal_id below.\n\n"
        f"{body}"
    )
