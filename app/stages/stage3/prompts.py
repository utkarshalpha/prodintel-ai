"""Prompt package and tool identity for Stage 3 (Conflict Detection)."""

from __future__ import annotations

from typing import Sequence
from uuid import UUID

from pydantic import BaseModel

__all__ = [
    "STAGE3_PROMPT_VERSION",
    "STAGE3_TOOL_NAME",
    "STAGE3_TOOL_DESCRIPTION",
    "FeatureForPrompt",
    "SignalForConflict",
    "build_system_prompt",
    "build_user_prompt",
]

STAGE3_PROMPT_VERSION = "stage3-v1"
STAGE3_TOOL_NAME = "emit_conflicts"

STAGE3_TOOL_DESCRIPTION = (
    "Identify conflicts between stakeholders over the provided features. A conflict "
    "requires >= 2 stakeholders holding genuinely OPPOSING positions on the same "
    "subject feature. Classify each conflict as priority, risk, resource, or "
    "strategic, and assign a severity (1-5). Every position MUST cite the "
    "signal_ids that substantiate it (a subset of the inputs only), and subject_id "
    "MUST be one of the provided feature_ids. If stakeholders do not genuinely "
    "disagree, return an empty conflicts list -- do not invent conflicts."
)


class FeatureForPrompt(BaseModel):
    """Minimal view of a candidate-subject feature."""

    feature_id: UUID
    title: str
    jtbd: str


class SignalForConflict(BaseModel):
    """Minimal view of an analyzed signal used as conflict evidence."""

    signal_id: UUID
    stakeholder_type: str
    intent: str
    claims: list[str]


def build_system_prompt() -> str:
    """Static instructions for the conflict-detection stage."""

    return (
        "You are the Conflict Detection stage of an enterprise product-decision "
        "system. Given a set of product features and the stakeholder signals behind "
        f"them, you surface DISAGREEMENTS between stakeholders. Respond ONLY by "
        f"calling the {STAGE3_TOOL_NAME!r} tool; never reply with free text.\n\n"
        "Rules:\n"
        "1. A conflict requires >= 2 stakeholders with genuinely opposing positions "
        "on the SAME subject feature (e.g. advocate vs oppose, or advocate vs "
        "risk_flag). Do not report agreement as conflict.\n"
        "2. Classify each conflict as one of: priority, risk, resource, strategic. "
        "Assign a severity from 1 (minor) to 5 (severe).\n"
        "3. For each stakeholder position, give a one-line summary and cite the "
        "signal_ids that substantiate it. Use only the provided signal_ids.\n"
        "4. subject_id must be one of the provided feature_ids.\n"
        "5. Provide a confidence object per conflict (score in [0, 1] + components).\n"
        "6. If the stakeholders do not genuinely disagree, return an empty "
        "conflicts list.\n"
        "A downstream validator re-checks that every cited id is real, that the "
        "positions truly oppose, and rejects the output otherwise."
    )


def build_user_prompt(
    features: Sequence[FeatureForPrompt],
    signals: Sequence[SignalForConflict],
) -> str:
    """Render the features (subjects) and signals (evidence) for conflict detection.

    Each feature and signal carries an explicit ``feature_id:`` / ``signal_id:`` line
    so the model and the integrity validator share unambiguous identifiers.
    """

    feature_blocks = "\n".join(
        f"[F{position}] feature_id: {feature.feature_id} | title: {feature.title}\n"
        f"    jtbd: {feature.jtbd}"
        for position, feature in enumerate(features, start=1)
    )
    signal_blocks = []
    for position, signal in enumerate(signals, start=1):
        claim_lines = "\n".join(f"      - {claim}" for claim in signal.claims) or "      - (none)"
        signal_blocks.append(
            f"[S{position}] signal_id: {signal.signal_id} | stakeholder: {signal.stakeholder_type}\n"
            f"    intent: {signal.intent}\n"
            f"    claims:\n{claim_lines}"
        )
    return (
        "Detect conflicts between stakeholders over the following features. "
        "Cite only the signal_ids listed, and use only these feature_ids as "
        "subjects.\n\n"
        f"Features:\n{feature_blocks}\n\n"
        f"Stakeholder signals:\n" + "\n".join(signal_blocks)
    )
