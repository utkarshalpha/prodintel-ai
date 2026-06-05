"""Stage 1 -- Signal Analysis contract.

This module defines the structured output the model is forced to produce when it
analyzes **one** immutable stakeholder signal. The model never replies with free
text; it must call a single tool whose input schema is exactly
:meth:`ParsedSignalContract.model_json_schema`.

The two objects here are pure *structural* contracts. They validate everything that
can be checked without the original signal text (ranges, span ordering, required
fields). The crucial **semantic** check -- that each claim's character span actually
substantiates the claim -- requires the raw signal text and therefore lives in
:mod:`app.ai_contracts.validation.grounding`. Keeping the two concerns separate is
deliberate: the contract guarantees shape, the grounding gate guarantees truth.
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from pydantic import Field, field_validator

from app.ai_contracts.base import AIContractBase, ConfidenceBlock, FrozenModel
from app.ai_contracts.enums import StakeholderType

__all__ = ["ExtractedClaim", "ParsedSignalContract"]


class ExtractedClaim(FrozenModel):
    """A single factual claim the model extracted from a signal, with its source.

    ``source_span`` is a half-open ``[start, end)`` character range into the
    signal's ``raw_text``. It is the anchor that makes the claim *traceable*: the
    grounding gate slices ``raw_text[start:end]`` and verifies it substantiates
    ``text``. A claim whose span does not support it is treated as a hallucination
    and dropped.

    This object validates only what it can see in isolation: non-empty text, a
    well-ordered span with a non-negative start, and a confidence in ``[0, 1]``.
    Whether ``end`` actually falls within a particular ``raw_text`` is checked by
    the grounding gate, which has the text.
    """

    text: str = Field(
        ...,
        min_length=1,
        description="The claim, paraphrased minimally from the source text.",
    )
    source_span: tuple[int, int] = Field(
        ...,
        description="Half-open [start, end) character offsets into the signal's raw_text.",
    )
    claim_confidence: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        ...,
        description="Model's confidence that this claim is correctly extracted, in [0, 1].",
    )

    @field_validator("source_span")
    @classmethod
    def _span_is_well_ordered(cls, value: tuple[int, int]) -> tuple[int, int]:
        """Ensure the span is internally valid: ``0 <= start < end``.

        Bounds against a specific ``raw_text`` length are *not* checked here (the
        text is not available at this layer); that is the grounding gate's job.
        """

        start, end = value
        if start < 0:
            raise ValueError(f"source_span start must be >= 0, got {start}")
        if end <= start:
            raise ValueError(
                f"source_span end ({end}) must be strictly greater than start ({start})"
            )
        return value


class ParsedSignalContract(AIContractBase):
    """Structured analysis of one stakeholder signal -- the Stage 1 tool output.

    Echoes ``signal_id`` back so a batch caller can detect a model that confuses
    one signal for another. Carries at least one :class:`ExtractedClaim`; a parse
    with zero claims is meaningless and is rejected at construction.

    Inherits ``schema_version`` and ``model_meta`` from :class:`AIContractBase`.
    """

    signal_id: UUID = Field(
        ...,
        description="ID of the analyzed signal, echoed from the input for cross-checking.",
    )
    intent: str = Field(
        ...,
        min_length=1,
        description="Single-sentence normalized statement of what the stakeholder wants.",
    )
    stakeholder_type: StakeholderType = Field(
        ...,
        description="Classified origin/role of the stakeholder behind this signal.",
    )
    urgency: Annotated[int, Field(ge=1, le=5)] = Field(
        ...,
        description="Urgency on a 1 (lowest) to 5 (highest) integer scale.",
    )
    sentiment: Annotated[float, Field(ge=-1.0, le=1.0)] = Field(
        ...,
        description="Sentiment from -1.0 (strongly negative) to 1.0 (strongly positive).",
    )
    extracted_claims: list[ExtractedClaim] = Field(
        ...,
        min_length=1,
        description="At least one source-anchored claim extracted from the signal.",
    )
    confidence: ConfidenceBlock = Field(
        ...,
        description="Decomposed confidence for the overall parse.",
    )
