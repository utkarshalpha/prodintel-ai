"""Stage 1 runner: Signal Analysis on top of the shared harness.

``Stage1SignalRunner`` subclasses :class:`~app.ai_runtime.base_runner.BaseStageRunner`
and supplies only Stage 1's identity, prompts, and the grounding validator. The
orchestration loop (call -> tool-check -> schema -> grounding -> retry -> result)
is entirely inherited.

Harness-managed metadata
------------------------
``ParsedSignalContract`` requires ``model_meta`` (provenance of the model call) and
carries a defaulted ``schema_version``. Neither should be produced by the model:
token counts and stop reasons are facts the *harness* observes, not the model. So
this runner:

* strips ``model_meta``/``schema_version`` from the tool schema the model sees
  (:meth:`tool_spec`), and
* injects an accurate ``model_meta`` -- built from the real
  :class:`~app.ai_runtime.interfaces.LLMToolResponse` -- before schema validation
  (:meth:`_schema_validate`).

The current response is captured in :meth:`_check_tool_call`, which the base loop
calls immediately before schema validation. As a result an instance processes one
signal at a time; create a runner per concurrent call (the
:func:`build_stage1_runner` factory makes that cheap).
"""

from __future__ import annotations

from typing import Callable
from uuid import UUID

from pydantic import Field

from app.ai_contracts.enums import StakeholderType
from app.ai_contracts.stage1_signal import ParsedSignalContract
from app.ai_contracts.validation.grounding import DEFAULT_CLAIM_THRESHOLD
from app.ai_runtime.base_runner import BaseStageRunner
from app.ai_runtime.interfaces import LLMToolResponse, StageContext, ToolSpec
from app.ai_runtime.retry_policy import RetryPolicy
from app.ai_runtime.validation_result import ValidationResult
from app.stages.stage1 import prompts
from app.stages.stage1.validators import GroundingValidator

__all__ = ["Stage1Context", "Stage1SignalRunner", "build_stage1_runner"]

#: Fields the harness owns and injects; never requested from the model.
_HARNESS_MANAGED_FIELDS = ("model_meta", "schema_version")


class Stage1Context(StageContext):
    """Inputs for one Stage 1 run: the signal to analyze."""

    signal_id: UUID = Field(..., description="ID of the signal under analysis.")
    raw_text: str = Field(..., min_length=1, description="Immutable text of the signal.")
    source_type: StakeholderType | None = Field(
        default=None,
        description="Ingestion channel, passed to the prompt as a hint.",
    )
    stage_name: str = Field(default="stage1")


class Stage1SignalRunner(BaseStageRunner[ParsedSignalContract, Stage1Context]):
    """Concrete runner for Stage 1 (Signal Analysis)."""

    def __init__(
        self,
        client,
        retry_policy: RetryPolicy | None = None,
        *,
        grounding_validator: GroundingValidator | None = None,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        extra: dict[str, object] = {}
        if sleep is not None:
            extra["sleep"] = sleep
        if clock is not None:
            extra["clock"] = clock
        super().__init__(client, retry_policy, **extra)  # type: ignore[arg-type]
        self._grounding = grounding_validator or GroundingValidator()
        self._pending_response: LLMToolResponse | None = None

    # --------------------------------------------------------- stage identity
    @property
    def stage_name(self) -> str:
        return "stage1"

    @property
    def tool_name(self) -> str:
        return prompts.STAGE1_TOOL_NAME

    @property
    def tool_description(self) -> str:
        return prompts.STAGE1_TOOL_DESCRIPTION

    def contract_model(self) -> type[ParsedSignalContract]:
        return ParsedSignalContract

    # ------------------------------------------------------------------ prompts
    def build_system_prompt(self, context: Stage1Context) -> str:
        return prompts.build_system_prompt()

    def build_user_prompt(self, context: Stage1Context) -> str:
        return prompts.build_user_prompt(
            signal_id=context.signal_id,
            raw_text=context.raw_text,
            source_type=context.source_type,
        )

    # ----------------------------------------------------------- semantic gates
    def semantic_validators(self):
        return (self._grounding,)

    # -------------------------------------------------- harness-managed metadata
    def tool_spec(self) -> ToolSpec:
        """Tool schema with harness-managed fields removed.

        The model is never asked for ``model_meta``/``schema_version``; those are
        injected by the runner from the real response before validation.
        """

        schema = self.contract_model().model_json_schema()
        properties = {
            name: spec
            for name, spec in schema.get("properties", {}).items()
            if name not in _HARNESS_MANAGED_FIELDS
        }
        required = [r for r in schema.get("required", []) if r not in _HARNESS_MANAGED_FIELDS]
        trimmed = {**schema, "properties": properties, "required": required}
        return ToolSpec(
            name=self.tool_name,
            description=self.tool_description,
            input_schema=trimmed,
        )

    def _check_tool_call(self, response: LLMToolResponse, attempt: int):
        """Capture the current response so :meth:`_schema_validate` can build
        accurate ``model_meta``, then defer to the base check."""

        self._pending_response = response
        return super()._check_tool_call(response, attempt)

    def _schema_validate(self, tool_input: dict) -> tuple[ParsedSignalContract | None, ValidationResult]:
        """Inject harness-managed ``model_meta`` before validating the tool input."""

        enriched = dict(tool_input)
        response = self._pending_response
        enriched["model_meta"] = {
            "model_id": response.model_id if response else "unknown",
            "prompt_version": prompts.STAGE1_PROMPT_VERSION,
            "input_tokens": response.input_tokens if response else 0,
            "output_tokens": response.output_tokens if response else 0,
            "stop_reason": response.stop_reason if response else "tool_use",
        }
        return super()._schema_validate(enriched)


def build_stage1_runner(
    client,
    *,
    retry_policy: RetryPolicy | None = None,
    grounding_threshold: float = DEFAULT_CLAIM_THRESHOLD,
    sleep: Callable[[float], None] | None = None,
    clock: Callable[[], float] | None = None,
) -> Stage1SignalRunner:
    """Dependency-injection factory for a configured :class:`Stage1SignalRunner`.

    Wires the grounding validator (at ``grounding_threshold``) and forwards the
    client, retry policy, and optional sleep/clock. This is the single place the
    application/service layer constructs a Stage 1 runner.
    """

    return Stage1SignalRunner(
        client,
        retry_policy,
        grounding_validator=GroundingValidator(grounding_threshold),
        sleep=sleep,
        clock=clock,
    )
