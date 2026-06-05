"""Prompt package and tool identity for Stage 1 (Signal Analysis).

Pure string builders plus the tool name/description and a pinned prompt version.
Kept dependency-free (no contract/runner imports) so prompts can be unit-tested and
versioned in isolation. The prompt version is stamped into every result's
``model_meta`` for reproducibility.
"""

from __future__ import annotations

from uuid import UUID

from app.ai_contracts.enums import StakeholderType

__all__ = [
    "STAGE1_PROMPT_VERSION",
    "STAGE1_TOOL_NAME",
    "STAGE1_TOOL_DESCRIPTION",
    "build_system_prompt",
    "build_user_prompt",
]

#: Bump when the wording below changes materially; recorded in model_meta.
STAGE1_PROMPT_VERSION = "stage1-v1"

#: Name of the tool the model must call. Mirrors the finalized AI-contract spec.
STAGE1_TOOL_NAME = "emit_parsed_signal"

STAGE1_TOOL_DESCRIPTION = (
    "Return the structured analysis of ONE stakeholder signal. Every claim's "
    "source_span MUST be exact, half-open [start, end) character offsets into the "
    "provided signal text. Do not paraphrase beyond what the text supports. If you "
    "cannot anchor a claim to a span that contains words substantiating it, omit "
    "the claim entirely."
)


def build_system_prompt() -> str:
    """Static system instructions for the Stage 1 analyst.

    Carries the rules that the deterministic grounding gate will later enforce, so
    a well-behaved model produces grounded output on the first attempt.
    """

    return (
        "You are the Signal Analysis stage of an enterprise product-decision system. "
        "You analyze exactly one stakeholder signal and respond ONLY by calling the "
        f"{STAGE1_TOOL_NAME!r} tool. Never reply with free text.\n\n"
        "Rules:\n"
        "1. Extract one or more factual claims that the signal actually states.\n"
        "2. For each claim, set source_span to the exact half-open [start, end) "
        "character offsets into the signal text whose substring contains the words "
        "that substantiate the claim. Offsets are 0-based; end is exclusive.\n"
        "3. Do not invent claims. If the text does not support a claim, leave it out.\n"
        "4. Classify stakeholder_type, an integer urgency from 1 (low) to 5 (high), "
        "and a sentiment from -1.0 (very negative) to 1.0 (very positive).\n"
        "5. Echo the signal_id you were given, unchanged.\n"
        "6. Provide a confidence object: a score in [0, 1] and a components map that "
        "includes 'span_grounding_ratio' (your estimate of the fraction of claims "
        "whose spans are exact).\n"
        "A downstream validator will re-check every span against the original text "
        "and reject any claim it cannot verify, so be precise rather than expansive."
    )


def build_user_prompt(
    *,
    signal_id: UUID,
    raw_text: str,
    source_type: StakeholderType | None = None,
) -> str:
    """Build the per-signal user prompt.

    The signal text is delimited so character offsets are unambiguous: offset 0 is
    the first character after the opening marker line.

    Parameters
    ----------
    signal_id:
        ID to echo back in the contract.
    raw_text:
        The immutable signal text to analyze and anchor spans into.
    source_type:
        The ingestion channel, supplied as a hint; the model still classifies
        stakeholder_type itself.
    """

    channel = f"\nKnown source channel (hint): {source_type.value}" if source_type else ""
    return (
        f"signal_id: {signal_id}{channel}\n"
        "Analyze the signal delimited below. Character offsets for source_span are "
        "measured from the first character of the signal text (the line after "
        "<<<SIGNAL>>>).\n\n"
        f"<<<SIGNAL>>>\n{raw_text}\n<<<END>>>"
    )
