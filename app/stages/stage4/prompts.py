"""Prompt package and tool identity for Stage 4 (Decision Synthesis)."""

from __future__ import annotations

from typing import Sequence
from uuid import UUID

from pydantic import BaseModel

__all__ = [
    "STAGE4_PROMPT_VERSION",
    "STAGE4_TOOL_NAME",
    "STAGE4_TOOL_DESCRIPTION",
    "FeatureForDecision",
    "SignalForDecision",
    "ConflictForDecision",
    "build_system_prompt",
    "build_user_prompt",
]

STAGE4_PROMPT_VERSION = "stage4-v1"
STAGE4_TOOL_NAME = "emit_decisions"

STAGE4_TOOL_DESCRIPTION = (
    "Synthesize evidence-backed product decisions over the provided features. Emit at "
    "least one decision. Each decision MUST be about one of the provided feature_ids "
    "(subject_id), recommend an action (build_now, build_later, reject, "
    "needs_discussion), give a rationale, and assign a priority_rank (1 = highest). "
    "Every decision MUST cite the signal_ids that substantiate it (a subset of the "
    "inputs only). If conflicts were detected over the subject feature, the decision "
    "MUST acknowledge every one of those conflict_ids and address them in the "
    "rationale -- it may not ignore a known conflict. Do not decide the same feature "
    "twice."
)


class FeatureForDecision(BaseModel):
    """Minimal view of a candidate-subject feature."""

    feature_id: UUID
    title: str
    jtbd: str


class SignalForDecision(BaseModel):
    """Minimal view of an analyzed signal used as decision evidence."""

    signal_id: UUID
    stakeholder_type: str
    intent: str
    claims: list[str]


class ConflictForDecision(BaseModel):
    """Minimal view of a detected conflict bearing on a subject feature."""

    conflict_id: UUID
    subject_id: UUID
    conflict_type: str
    severity: int
    summary: str


def build_system_prompt() -> str:
    """Static instructions for the decision-synthesis stage."""

    return (
        "You are the Decision Synthesis stage of an enterprise product-decision "
        "system. Given a set of product features, the stakeholder signals behind "
        "them, and the conflicts already detected over them, you produce defensible, "
        f"evidence-backed product decisions. Respond ONLY by calling the "
        f"{STAGE4_TOOL_NAME!r} tool; never reply with free text.\n\n"
        "Rules:\n"
        "1. Emit at least one decision. subject_id must be one of the provided "
        "feature_ids, and you must not decide the same feature twice.\n"
        "2. Recommend exactly one action per decision: build_now, build_later, "
        "reject, or needs_discussion. Assign a priority_rank (1 = highest priority).\n"
        "3. Give a rationale that references the evidence and explicitly addresses "
        "any conflicts over the subject feature.\n"
        "4. Cite the signal_ids that substantiate the decision. Use only the provided "
        "signal_ids.\n"
        "5. If conflicts were detected over the subject feature, acknowledge EVERY "
        "one of those conflict_ids. You may not ignore a known conflict.\n"
        "6. Provide a confidence object per decision (score in [0, 1] + components).\n"
        "A downstream validator re-checks that every cited id is real, that no "
        "conflict over the subject is left unacknowledged, and rejects the output "
        "otherwise."
    )


def build_user_prompt(
    features: Sequence[FeatureForDecision],
    signals: Sequence[SignalForDecision],
    conflicts: Sequence[ConflictForDecision],
) -> str:
    """Render the features (subjects), signals (evidence), and conflicts (to address).

    Each row carries an explicit ``feature_id:`` / ``signal_id:`` / ``conflict_id:``
    line so the model and the integrity validator share unambiguous identifiers.
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

    if conflicts:
        conflict_blocks = "\n".join(
            f"[C{position}] conflict_id: {conflict.conflict_id} | subject_id: {conflict.subject_id} "
            f"| type: {conflict.conflict_type} | severity: {conflict.severity}\n"
            f"    summary: {conflict.summary}"
            for position, conflict in enumerate(conflicts, start=1)
        )
    else:
        conflict_blocks = "(none detected)"

    return (
        "Synthesize evidence-backed decisions over the following features. Cite only "
        "the signal_ids listed, use only these feature_ids as subjects, and "
        "acknowledge every conflict_id whose subject_id matches the feature you are "
        "deciding.\n\n"
        f"Features:\n{feature_blocks}\n\n"
        f"Stakeholder signals:\n" + "\n".join(signal_blocks) + "\n\n"
        f"Detected conflicts:\n{conflict_blocks}"
    )
