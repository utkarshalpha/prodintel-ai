"""Stage 4 runner: Decision Synthesis on the shared harness.

``Stage4DecisionRunner`` subclasses :class:`~app.ai_runtime.base_runner.BaseStageRunner`
and mixes in :class:`~app.stages._harness_metadata.HarnessManagedMetadataRunner`
(so harness-managed ``model_meta`` is injected from the real response without
re-copying that logic). It supplies Stage 4's identity, prompts, and the decision
integrity validator; the orchestration loop is inherited.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence
from uuid import UUID

from pydantic import Field

from app.ai_contracts.stage4_decision import DecisionSynthesisContract
from app.ai_runtime.base_runner import BaseStageRunner
from app.ai_runtime.interfaces import LLMToolResponse, StageContext
from app.ai_runtime.retry_policy import RetryPolicy
from app.stages._harness_metadata import HarnessManagedMetadataRunner
from app.stages.stage4 import prompts
from app.stages.stage4.prompts import ConflictForDecision, FeatureForDecision, SignalForDecision
from app.stages.stage4.validators import DecisionIntegrityValidator

__all__ = ["Stage4Context", "Stage4DecisionRunner", "build_stage4_runner"]


class Stage4Context(StageContext):
    """Inputs for one synthesis run: the features, their signals, and their conflicts."""

    features: list[FeatureForDecision] = Field(..., min_length=1, description="Subject features to decide on.")
    signals: list[SignalForDecision] = Field(..., min_length=1, description="Analyzed signals (evidence).")
    conflicts: list[ConflictForDecision] = Field(
        default_factory=list,
        description="Conflicts already detected over the features (possibly none).",
    )
    workspace_id: UUID | None = Field(default=None, description="Optional workspace scope (reserved).")
    stage_name: str = Field(default="stage4")

    @property
    def input_feature_ids(self) -> list[UUID]:
        """Valid decision subjects (the features submitted for synthesis)."""

        return [feature.feature_id for feature in self.features]

    @property
    def input_signal_ids(self) -> list[UUID]:
        """Valid evidence ids (the signals behind the features)."""

        return [signal.signal_id for signal in self.signals]

    @property
    def conflict_subjects(self) -> Mapping[UUID, UUID]:
        """Map each input conflict id to the feature id it is about."""

        return {conflict.conflict_id: conflict.subject_id for conflict in self.conflicts}


class Stage4DecisionRunner(HarnessManagedMetadataRunner, BaseStageRunner[DecisionSynthesisContract, Stage4Context]):
    """Concrete runner for Stage 4 (Decision Synthesis)."""

    def __init__(
        self,
        client,
        retry_policy: RetryPolicy | None = None,
        *,
        integrity_validator: DecisionIntegrityValidator | None = None,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        extra: dict[str, object] = {}
        if sleep is not None:
            extra["sleep"] = sleep
        if clock is not None:
            extra["clock"] = clock
        super().__init__(client, retry_policy, **extra)  # type: ignore[arg-type]
        self._integrity = integrity_validator or DecisionIntegrityValidator()
        self._pending_response: LLMToolResponse | None = None

    # --------------------------------------------------------- stage identity
    @property
    def stage_name(self) -> str:
        return "stage4"

    @property
    def tool_name(self) -> str:
        return prompts.STAGE4_TOOL_NAME

    @property
    def tool_description(self) -> str:
        return prompts.STAGE4_TOOL_DESCRIPTION

    @property
    def prompt_version(self) -> str:
        return prompts.STAGE4_PROMPT_VERSION

    def contract_model(self) -> type[DecisionSynthesisContract]:
        return DecisionSynthesisContract

    # ------------------------------------------------------------------ prompts
    def build_system_prompt(self, context: Stage4Context) -> str:
        return prompts.build_system_prompt()

    def build_user_prompt(self, context: Stage4Context) -> str:
        return prompts.build_user_prompt(context.features, context.signals, context.conflicts)

    # ----------------------------------------------------------- semantic gates
    def semantic_validators(self) -> Sequence[DecisionIntegrityValidator]:
        return (self._integrity,)


def build_stage4_runner(
    client,
    *,
    retry_policy: RetryPolicy | None = None,
    sleep: Callable[[float], None] | None = None,
    clock: Callable[[], float] | None = None,
) -> Stage4DecisionRunner:
    """Dependency-injection factory for a configured :class:`Stage4DecisionRunner`."""

    return Stage4DecisionRunner(client, retry_policy, sleep=sleep, clock=clock)
