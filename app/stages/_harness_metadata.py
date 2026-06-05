"""Shared mixin for harness-managed contract metadata.

Stages 1 and 2 each strip ``model_meta``/``schema_version`` from the tool the model
sees and inject an accurate ``model_meta`` (token counts, stop reason) from the real
response before schema validation -- so the model never fabricates call provenance.
This mixin factors that behavior out so new stages do not re-copy it.

It overrides three :class:`~app.ai_runtime.base_runner.BaseStageRunner` extension
points and delegates to ``super()`` for the base behavior, so it must appear *before*
``BaseStageRunner`` in a runner's MRO::

    class Stage3ConflictRunner(HarnessManagedMetadataRunner, BaseStageRunner[...]):
        ...

Concrete runners must provide :attr:`prompt_version`. The mixin is deliberately not
applied retroactively to Stages 1 and 2 (which are frozen and passing); it exists so
Stage 3 adds no further duplication.
"""

from __future__ import annotations

from app.ai_runtime.interfaces import LLMToolResponse, ToolSpec

__all__ = ["HarnessManagedMetadataRunner"]

_HARNESS_MANAGED_FIELDS = ("model_meta", "schema_version")


class HarnessManagedMetadataRunner:
    """Mixin that hides harness-managed fields from the model and injects them."""

    @property
    def prompt_version(self) -> str:
        """Prompt version recorded in the injected ``model_meta``. Must be overridden."""

        raise NotImplementedError

    def tool_spec(self) -> ToolSpec:
        """Build the tool schema with harness-managed fields removed."""

        schema = self.contract_model().model_json_schema()  # type: ignore[attr-defined]
        properties = {
            name: spec
            for name, spec in schema.get("properties", {}).items()
            if name not in _HARNESS_MANAGED_FIELDS
        }
        required = [r for r in schema.get("required", []) if r not in _HARNESS_MANAGED_FIELDS]
        trimmed = {**schema, "properties": properties, "required": required}
        return ToolSpec(
            name=self.tool_name,  # type: ignore[attr-defined]
            description=self.tool_description,  # type: ignore[attr-defined]
            input_schema=trimmed,
        )

    def _check_tool_call(self, response: LLMToolResponse, attempt: int):
        """Capture the current response so metadata can be injected, then defer."""

        self._pending_response = response
        return super()._check_tool_call(response, attempt)  # type: ignore[misc]

    def _schema_validate(self, tool_input: dict):
        """Inject harness-managed ``model_meta`` from the response before validating."""

        enriched = dict(tool_input)
        response: LLMToolResponse | None = getattr(self, "_pending_response", None)
        enriched["model_meta"] = {
            "model_id": response.model_id if response else "unknown",
            "prompt_version": self.prompt_version,
            "input_tokens": response.input_tokens if response else 0,
            "output_tokens": response.output_tokens if response else 0,
            "stop_reason": response.stop_reason if response else "tool_use",
        }
        return super()._schema_validate(enriched)  # type: ignore[misc]
