"""Stage 3 runner: Conflict Detection on the shared harness.

``Stage3ConflictRunner`` subclasses :class:`~app.ai_runtime.base_runner.BaseStageRunner`
and mixes in :class:`~app.stages._harness_metadata.HarnessManagedMetadataRunner`
(so harness-managed ``model_meta`` is injected from the real response without
re-copying that logic). It supplies Stage 3's identity, prompts, and the conflict
integrity validator; the orchestration loop is inherited.
"""

from __future__ import annotations

from typing import Callable, Sequence
from uuid import UUID

from pydantic import Field

from app.ai_contracts.stage3_conflict import ConflictDetectionContract
from app.ai_runtime.base_runner import BaseStageRunner
from app.ai_runtime.interfaces import LLMToolResponse, StageContext
from app.ai_runtime.retry_policy import RetryPolicy
from app.stages._harness_metadata import HarnessManagedMetadataRunner
from app.stages.stage3 import prompts
from app.stages.stage3.prompts import FeatureForPrompt, SignalForConflict
from app.stages.stage3.validators import ConflictIntegrityValidator

__all__ = ["Stage3Context", "Stage3ConflictRunner", "build_stage3_runner"]


class Stage3Context(StageContext):
    """Inputs for one conflict-detection run: the features and their signals."""

    features: list[FeatureForPrompt] = Field(..., min_length=1, description="Subject features under review.")
    signals: list[SignalForConflict] = Field(..., min_length=1, description="Analyzed signals (evidence).")
    workspace_id: UUID | None = Field(default=None, description="Optional workspace scope (reserved).")
    stage_name: str = Field(default="stage3")

    @property
    def input_feature_ids(self) -> list[UUID]:
        """Valid conflict subjects (the features submitted for detection)."""

        return [feature.feature_id for feature in self.features]

    @property
    def input_signal_ids(self) -> list[UUID]:
        """Valid evidence ids (the signals behind the features)."""

        return [signal.signal_id for signal in self.signals]


class Stage3ConflictRunner(HarnessManagedMetadataRunner, BaseStageRunner[ConflictDetectionContract, Stage3Context]):
    """Concrete runner for Stage 3 (Conflict Detection)."""

    def __init__(
        self,
        client,
        retry_policy: RetryPolicy | None = None,
        *,
        integrity_validator: ConflictIntegrityValidator | None = None,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        extra: dict[str, object] = {}
        if sleep is not None:
            extra["sleep"] = sleep
        if clock is not None:
            extra["clock"] = clock
        super().__init__(client, retry_policy, **extra)  # type: ignore[arg-type]
        self._integrity = integrity_validator or ConflictIntegrityValidator()
        self._pending_response: LLMToolResponse | None = None

    # --------------------------------------------------------- stage identity
    @property
    def stage_name(self) -> str:
        return "stage3"

    @property
    def tool_name(self) -> str:
        return prompts.STAGE3_TOOL_NAME

    @property
    def tool_description(self) -> str:
        return prompts.STAGE3_TOOL_DESCRIPTION

    @property
    def prompt_version(self) -> str:
        return prompts.STAGE3_PROMPT_VERSION

    def contract_model(self) -> type[ConflictDetectionContract]:
        return ConflictDetectionContract

    # ------------------------------------------------------------------ prompts
    def build_system_prompt(self, context: Stage3Context) -> str:
        return prompts.build_system_prompt()

    def build_user_prompt(self, context: Stage3Context) -> str:
        return prompts.build_user_prompt(context.features, context.signals)

    # ----------------------------------------------------------- semantic gates
    def semantic_validators(self) -> Sequence[ConflictIntegrityValidator]:
        return (self._integrity,)


def build_stage3_runner(
    client,
    *,
    retry_policy: RetryPolicy | None = None,
    sleep: Callable[[float], None] | None = None,
    clock: Callable[[], float] | None = None,
) -> Stage3ConflictRunner:
    """Dependency-injection factory for a configured :class:`Stage3ConflictRunner`."""

    return Stage3ConflictRunner(client, retry_policy, sleep=sleep, clock=clock)
