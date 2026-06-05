"""Stage 2 runner: Feature Extraction on the shared harness.

``Stage2FeatureRunner`` subclasses :class:`~app.ai_runtime.base_runner.BaseStageRunner`
and supplies Stage 2's identity, prompts, and the provenance validator. It applies
the same harness-managed-metadata approach as Stage 1: ``model_meta``/``schema_version``
are stripped from the tool the model sees and injected from the real response before
validation, so the model never fabricates provenance metadata.
"""

from __future__ import annotations

from typing import Callable, Sequence
from uuid import UUID

from pydantic import Field

from app.ai_contracts.stage2_feature import FeatureExtractionContract
from app.ai_runtime.base_runner import BaseStageRunner
from app.ai_runtime.interfaces import LLMToolResponse, StageContext, ToolSpec
from app.ai_runtime.retry_policy import RetryPolicy
from app.ai_runtime.validation_result import ValidationResult
from app.stages.stage2 import prompts
from app.stages.stage2.prompts import SignalForPrompt
from app.stages.stage2.validators import FeatureProvenanceValidator

__all__ = ["Stage2Context", "Stage2FeatureRunner", "build_stage2_runner"]

_HARNESS_MANAGED_FIELDS = ("model_meta", "schema_version")


class Stage2Context(StageContext):
    """Inputs for one extraction run: the analyzed signals to cluster."""

    signals: list[SignalForPrompt] = Field(..., min_length=1, description="Analyzed signals to cluster.")
    workspace_id: UUID | None = Field(default=None, description="Optional workspace scope (reserved).")
    stage_name: str = Field(default="stage2")

    @property
    def input_signal_ids(self) -> list[UUID]:
        """The universe of signal ids the provenance validator checks against."""

        return [signal.signal_id for signal in self.signals]


class Stage2FeatureRunner(BaseStageRunner[FeatureExtractionContract, Stage2Context]):
    """Concrete runner for Stage 2 (Feature Extraction)."""

    def __init__(
        self,
        client,
        retry_policy: RetryPolicy | None = None,
        *,
        provenance_validator: FeatureProvenanceValidator | None = None,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        extra: dict[str, object] = {}
        if sleep is not None:
            extra["sleep"] = sleep
        if clock is not None:
            extra["clock"] = clock
        super().__init__(client, retry_policy, **extra)  # type: ignore[arg-type]
        self._provenance = provenance_validator or FeatureProvenanceValidator()
        self._pending_response: LLMToolResponse | None = None

    # --------------------------------------------------------- stage identity
    @property
    def stage_name(self) -> str:
        return "stage2"

    @property
    def tool_name(self) -> str:
        return prompts.STAGE2_TOOL_NAME

    @property
    def tool_description(self) -> str:
        return prompts.STAGE2_TOOL_DESCRIPTION

    def contract_model(self) -> type[FeatureExtractionContract]:
        return FeatureExtractionContract

    # ------------------------------------------------------------------ prompts
    def build_system_prompt(self, context: Stage2Context) -> str:
        return prompts.build_system_prompt()

    def build_user_prompt(self, context: Stage2Context) -> str:
        return prompts.build_user_prompt(context.signals)

    # ----------------------------------------------------------- semantic gates
    def semantic_validators(self) -> Sequence[FeatureProvenanceValidator]:
        return (self._provenance,)

    # -------------------------------------------------- harness-managed metadata
    def tool_spec(self) -> ToolSpec:
        schema = self.contract_model().model_json_schema()
        properties = {
            name: spec
            for name, spec in schema.get("properties", {}).items()
            if name not in _HARNESS_MANAGED_FIELDS
        }
        required = [r for r in schema.get("required", []) if r not in _HARNESS_MANAGED_FIELDS]
        trimmed = {**schema, "properties": properties, "required": required}
        return ToolSpec(name=self.tool_name, description=self.tool_description, input_schema=trimmed)

    def _check_tool_call(self, response: LLMToolResponse, attempt: int):
        self._pending_response = response
        return super()._check_tool_call(response, attempt)

    def _schema_validate(self, tool_input: dict) -> tuple[FeatureExtractionContract | None, ValidationResult]:
        enriched = dict(tool_input)
        response = self._pending_response
        enriched["model_meta"] = {
            "model_id": response.model_id if response else "unknown",
            "prompt_version": prompts.STAGE2_PROMPT_VERSION,
            "input_tokens": response.input_tokens if response else 0,
            "output_tokens": response.output_tokens if response else 0,
            "stop_reason": response.stop_reason if response else "tool_use",
        }
        return super()._schema_validate(enriched)


def build_stage2_runner(
    client,
    *,
    retry_policy: RetryPolicy | None = None,
    sleep: Callable[[float], None] | None = None,
    clock: Callable[[], float] | None = None,
) -> Stage2FeatureRunner:
    """Dependency-injection factory for a configured :class:`Stage2FeatureRunner`."""

    return Stage2FeatureRunner(client, retry_policy, sleep=sleep, clock=clock)
