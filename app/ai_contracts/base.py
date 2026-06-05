"""Foundational Pydantic v2 building blocks for every AI-stage contract.

This module defines the *trust boundary* primitives shared by all four LLM stages
of ProdIntel AI:

* :class:`FrozenModel` -- the strict, immutable base for every contract object.
* :class:`ModelMeta` -- provenance about the model call that produced a contract.
* :class:`ConfidenceBlock` -- a decomposed, self-consistent confidence value.
* :class:`AIContractBase` -- the base for top-level contract roots (carries
  ``schema_version`` + ``model_meta``).

Design rules enforced here (and relied upon by the rest of the system):

1. **Reject the unexpected.** ``extra="forbid"`` means an LLM that invents a field
   causes a hard validation error rather than silently polluting state.
2. **Immutability.** ``frozen=True`` means a validated contract cannot be mutated
   after construction, so a downstream stage can never tamper with an upstream
   stage's output.
3. **Confidence is computed, not asserted.** The qualitative
   :class:`ConfidenceBasis` band is always derived from the numeric score, so the
   label and the number cannot drift apart -- even if the model emits a
   contradictory band.

Nested value objects (:class:`ModelMeta`, :class:`ConfidenceBlock`, and the
per-stage claim objects) inherit from :class:`FrozenModel` rather than
:class:`AIContractBase`: only the *root* contract a model emits needs to carry the
schema version and call metadata, so embedding that on every nested object would be
redundant and would force the model to repeat it.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.ai_contracts.enums import ConfidenceBasis

__all__ = [
    "FrozenModel",
    "ModelMeta",
    "ConfidenceBlock",
    "AIContractBase",
    "confidence_basis_for_score",
]


# Score band thresholds. Defined once here so the database, the API layer, and the
# UI can all reference the same cut points. Ordered high -> low.
_STRONG_FLOOR = 0.8
_MODERATE_FLOOR = 0.6
_WEAK_FLOOR = 0.4


def confidence_basis_for_score(score: float) -> ConfidenceBasis:
    """Map a numeric confidence score in ``[0, 1]`` to a qualitative band.

    This is the **single** place the score-to-band mapping lives. The bands are:

    * ``>= 0.8`` -> :attr:`~app.ai_contracts.enums.ConfidenceBasis.STRONG`
    * ``>= 0.6`` -> :attr:`~app.ai_contracts.enums.ConfidenceBasis.MODERATE`
    * ``>= 0.4`` -> :attr:`~app.ai_contracts.enums.ConfidenceBasis.WEAK`
    * ``<  0.4`` -> :attr:`~app.ai_contracts.enums.ConfidenceBasis.INSUFFICIENT`

    Parameters
    ----------
    score:
        Confidence score. Callers are responsible for range validation; this
        function clamps defensively so it never raises on out-of-range input.

    Returns
    -------
    ConfidenceBasis
        The band corresponding to ``score``.
    """

    # Defensive clamp: a band must always be returnable even for a malformed score.
    clamped = min(1.0, max(0.0, float(score)))
    if clamped >= _STRONG_FLOOR:
        return ConfidenceBasis.STRONG
    if clamped >= _MODERATE_FLOOR:
        return ConfidenceBasis.MODERATE
    if clamped >= _WEAK_FLOOR:
        return ConfidenceBasis.WEAK
    return ConfidenceBasis.INSUFFICIENT


class FrozenModel(BaseModel):
    """Strict, immutable base for every AI-contract object.

    * ``extra="forbid"`` -- unknown keys raise a ``ValidationError``.
    * ``frozen=True`` -- instances are hashable and cannot be mutated.
    * ``validate_assignment=True`` -- defensive; assignment is already blocked by
      ``frozen`` but this makes intent explicit.
    * ``str_strip_whitespace=True`` -- leading/trailing whitespace on strings is
      normalized away so "  " never sneaks past a ``min_length`` check.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_assignment=True,
        str_strip_whitespace=True,
    )


class ModelMeta(FrozenModel):
    """Provenance describing the single model call that produced a contract.

    Captured so that every artifact in the platform is traceable not only to its
    *inputs* (via the evidence graph) but also to the exact model invocation that
    generated it -- essential for the research paper's reproducibility story and
    for debugging regressions when a prompt or model version changes.
    """

    model_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the model that produced the contract, e.g. 'claude-opus-4-8'.",
    )
    prompt_version: str = Field(
        ...,
        min_length=1,
        description="Version tag of the prompt template used for this stage.",
    )
    input_tokens: int = Field(
        ...,
        ge=0,
        description="Prompt token count reported by the API for this call.",
    )
    output_tokens: int = Field(
        ...,
        ge=0,
        description="Completion token count reported by the API for this call.",
    )
    stop_reason: str = Field(
        ...,
        min_length=1,
        description="Stop reason reported by the API, e.g. 'tool_use', 'end_turn', 'max_tokens'.",
    )


class ConfidenceBlock(FrozenModel):
    """A decomposed, internally consistent confidence value.

    A confidence is never a bare float in this system. It carries:

    * ``score`` -- the headline value in ``[0, 1]``.
    * ``components`` -- the named sub-scores that justify it (stage-specific,
      e.g. ``span_grounding_ratio``), each itself in ``[0, 1]``.
    * ``basis`` -- the qualitative band, **always recomputed** from ``score`` so it
      cannot contradict the number.

    The ``basis`` is derived in a ``mode="before"`` validator: whatever value (if
    any) is supplied for ``basis`` is overwritten with the band computed from
    ``score``. This means a model can emit a confidence block and never lie about
    the band -- the contract self-corrects it.
    """

    score: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        ...,
        description="Headline confidence in [0, 1].",
    )
    components: dict[str, float] = Field(
        default_factory=dict,
        description="Named sub-scores justifying the headline, each in [0, 1].",
    )
    basis: ConfidenceBasis = Field(
        ...,
        description="Qualitative band, always derived from `score` (never trusted from input).",
    )

    @model_validator(mode="before")
    @classmethod
    def _derive_basis_from_score(cls, data: Any) -> Any:
        """Overwrite (or inject) ``basis`` so it always matches ``score``.

        Runs before field validation. If ``score`` is present and coercible, the
        ``basis`` key is set to the recomputed band, discarding any value the
        caller/model supplied. If ``score`` is missing or non-numeric, the input is
        passed through untouched so the normal ``score`` validation produces the
        canonical error message.
        """

        if isinstance(data, dict) and "score" in data:
            try:
                score = float(data["score"])
            except (TypeError, ValueError):
                return data
            # Build a new dict so we never mutate the caller's object.
            return {**data, "basis": confidence_basis_for_score(score)}
        return data

    @field_validator("components")
    @classmethod
    def _components_in_unit_interval(cls, value: dict[str, float]) -> dict[str, float]:
        """Reject any component sub-score outside ``[0, 1]``."""

        for name, component in value.items():
            if not 0.0 <= float(component) <= 1.0:
                raise ValueError(
                    f"confidence component {name!r} must be in [0, 1], got {component!r}"
                )
        return value


class AIContractBase(FrozenModel):
    """Base for **top-level** contract roots emitted by a model stage.

    Adds the two fields every stage output must carry:

    * ``schema_version`` -- pinned literal so a future schema change is an explicit,
      detectable migration rather than a silent drift.
    * ``model_meta`` -- provenance of the producing model call.

    Nested value objects do **not** inherit from this class (see module docstring).
    """

    schema_version: Literal["1.0"] = Field(
        default="1.0",
        description="Contract schema version. Pinned so changes are explicit migrations.",
    )
    model_meta: ModelMeta = Field(
        ...,
        description="Provenance of the model call that produced this contract.",
    )
