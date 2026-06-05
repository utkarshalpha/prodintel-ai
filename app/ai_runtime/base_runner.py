"""Abstract stage runner: the one orchestration loop all four stages share.

A concrete stage supplies only its *identity* and *prompts*:

* :meth:`BaseStageRunner.contract_model` -- the Pydantic contract to validate against.
* :attr:`tool_name` / :attr:`tool_description` -- how the tool is presented to the model.
* :meth:`build_system_prompt` / :meth:`build_user_prompt` -- the stage's instructions.
* :meth:`semantic_validators` -- optional stage-specific gates (e.g. grounding).

The base owns everything else, identically for every stage:

    call the model (tool-forced)
      -> was a usable tool call produced?      (else: tool-call error)
      -> schema-validate the tool input        (else: schema error + feedback)
      -> run semantic validators               (else: semantic error + feedback)
      -> success: collect confidence + metrics, return StageResult
    on any retryable failure: append deterministic feedback, consult RetryPolicy,
    and try again until success or the policy says stop.

The loop is fully deterministic given a deterministic client: no randomness, no
hidden clock dependence (latency measurement is optional and injected). This is what
lets the research harness replay a run exactly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Generic, Sequence, TypeVar

from pydantic import BaseModel, ValidationError

from app.ai_contracts.base import ConfidenceBlock
from app.ai_runtime.errors import ErrorCode, LLMClientError, StageError
from app.ai_runtime.interfaces import (
    LLMMessage,
    LLMToolResponse,
    StageContext,
    StageValidator,
    ToolCallClient,
    ToolSpec,
)
from app.ai_runtime.metrics import AttemptMetrics, StageMetrics
from app.ai_runtime.retry_policy import RetryPolicy
from app.ai_runtime.stage_result import StageResult, StageStatus
from app.ai_runtime.validation_result import ValidationResult

__all__ = ["BaseStageRunner"]

TContract = TypeVar("TContract", bound=BaseModel)
TContext = TypeVar("TContext", bound=StageContext)

# A no-op sleeper: with the default zero-delay RetryPolicy this is never invoked
# anyway, but injecting it keeps the runner free of an implicit time.sleep import
# and lets tests assert no real waiting occurs.
_NoOpSleep: Callable[[float], None] = lambda _seconds: None


class BaseStageRunner(ABC, Generic[TContract, TContext]):
    """Reusable execution engine for one AI stage.

    Parameters
    ----------
    client:
        The :class:`ToolCallClient` used to call the model. Injected so the runner
        is provider-agnostic and unit-testable with a scripted fake.
    retry_policy:
        The :class:`RetryPolicy` governing re-attempts. Defaults to the standard
        three-attempt, zero-delay policy.
    sleep:
        Callable invoked with a delay in seconds before a retry. Defaults to a
        no-op (tests never wait); production wires ``time.sleep``.
    clock:
        Optional zero-argument callable returning a monotonic time in seconds. When
        provided, per-attempt latency is recorded in metrics.
    """

    def __init__(
        self,
        client: ToolCallClient,
        retry_policy: RetryPolicy | None = None,
        *,
        sleep: Callable[[float], None] = _NoOpSleep,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._client = client
        self._retry_policy = retry_policy or RetryPolicy()
        self._sleep = sleep
        self._clock = clock

    # ----------------------------------------------------------- stage identity
    @property
    @abstractmethod
    def stage_name(self) -> str:
        """Stable identifier for this stage (used in metrics and results)."""

    @property
    @abstractmethod
    def tool_name(self) -> str:
        """Name of the tool the model must call."""

    @property
    @abstractmethod
    def tool_description(self) -> str:
        """Description presented to the model for the tool."""

    @abstractmethod
    def contract_model(self) -> type[TContract]:
        """The Pydantic contract class the tool input must validate against."""

    # ------------------------------------------------------------------ prompts
    @abstractmethod
    def build_system_prompt(self, context: TContext) -> str:
        """Build the system prompt for this run."""

    @abstractmethod
    def build_user_prompt(self, context: TContext) -> str:
        """Build the initial user prompt for this run."""

    # -------------------------------------------------------------- extensibility
    def semantic_validators(self) -> Sequence[StageValidator]:
        """Stage-specific gates run after schema validation. Default: none."""

        return ()

    def tool_spec(self) -> ToolSpec:
        """Build the tool spec from the contract's JSON schema.

        Generating the tool the model sees and the validator the harness runs from
        the same ``model_json_schema()`` guarantees they cannot disagree.
        """

        return ToolSpec(
            name=self.tool_name,
            description=self.tool_description,
            input_schema=self.contract_model().model_json_schema(),
        )

    def collect_confidence(self, contract: TContract) -> ConfidenceBlock | None:
        """Surface a contract's confidence block, if it exposes one."""

        candidate = getattr(contract, "confidence", None)
        return candidate if isinstance(candidate, ConfidenceBlock) else None

    # --------------------------------------------------------------------- run
    def run(self, context: TContext) -> StageResult[TContract]:
        """Execute the stage end-to-end, with validation and deterministic retries."""

        tool = self.tool_spec()
        system = self.build_system_prompt(context)
        messages: list[LLMMessage] = [LLMMessage(role="user", content=self.build_user_prompt(context))]

        metrics = StageMetrics(stage_name=self.stage_name)
        last_error: StageError | None = None
        last_validation: ValidationResult = ValidationResult.success()

        attempt = 0
        while True:
            attempt += 1

            # --- 1. Call the model, capturing transport failures. -------------
            started = self._clock() if self._clock else None
            try:
                response = self._client.complete(system=system, messages=messages, tool=tool)
            except LLMClientError as exc:
                error = StageError(
                    code=ErrorCode.CLIENT_ERROR,
                    message=f"client call failed: {exc}",
                    attempt=attempt,
                    recoverable=True,
                )
                metrics = metrics.with_attempt(
                    self._attempt_metrics(attempt, None, schema_valid=False, semantic_valid=False, error=error, started=started)
                )
                decision = self._retry_policy.decide(attempt=attempt, failure_code=error.code)
                if decision.should_retry:
                    self._wait(decision.delay_seconds)
                    continue
                return self._fail(metrics, error, last_validation, attempt, exhausted=decision.exhausted)

            # --- 2. Did the model produce a usable tool call? -----------------
            tool_error = self._check_tool_call(response, attempt)
            if tool_error is not None:
                last_error = tool_error
                metrics = metrics.with_attempt(
                    self._attempt_metrics(attempt, response, schema_valid=False, semantic_valid=False, error=tool_error, started=started)
                )
                messages.append(self._tool_call_correction(tool_error))
                decision = self._retry_policy.decide(attempt=attempt, failure_code=tool_error.code)
                if decision.should_retry:
                    self._wait(decision.delay_seconds)
                    continue
                return self._fail(metrics, tool_error, last_validation, attempt, exhausted=decision.exhausted)

            # --- 3. Schema validation. ----------------------------------------
            assert response.tool_input is not None  # guaranteed by _check_tool_call
            contract, schema_result = self._schema_validate(response.tool_input)
            if contract is None:
                last_validation = schema_result
                last_error = StageError(
                    code=ErrorCode.SCHEMA_VALIDATION_FAILED,
                    message="tool input failed schema validation",
                    attempt=attempt,
                    recoverable=True,
                    details={"issues": [i.model_dump() for i in schema_result.errors]},
                )
                metrics = metrics.with_attempt(
                    self._attempt_metrics(attempt, response, schema_valid=False, semantic_valid=False, error=last_error, started=started)
                )
                messages.append(LLMMessage(role="user", content=schema_result.retry_feedback()))
                decision = self._retry_policy.decide(attempt=attempt, failure_code=last_error.code)
                if decision.should_retry:
                    self._wait(decision.delay_seconds)
                    continue
                return self._fail(metrics, last_error, last_validation, attempt, exhausted=decision.exhausted)

            # --- 4. Semantic validation (stage-specific gates). ---------------
            semantic_result = self._semantic_validate(contract, context)
            if not semantic_result.ok:
                last_validation = semantic_result
                last_error = StageError(
                    code=ErrorCode.SEMANTIC_VALIDATION_FAILED,
                    message="contract failed semantic validation",
                    attempt=attempt,
                    recoverable=True,
                    details={"issues": [i.model_dump() for i in semantic_result.errors]},
                )
                metrics = metrics.with_attempt(
                    self._attempt_metrics(attempt, response, schema_valid=True, semantic_valid=False, error=last_error, started=started)
                )
                messages.append(LLMMessage(role="user", content=semantic_result.retry_feedback()))
                decision = self._retry_policy.decide(attempt=attempt, failure_code=last_error.code)
                if decision.should_retry:
                    self._wait(decision.delay_seconds)
                    continue
                return self._fail(metrics, last_error, last_validation, attempt, exhausted=decision.exhausted)

            # --- 5. Success. --------------------------------------------------
            metrics = metrics.with_attempt(
                self._attempt_metrics(attempt, response, schema_valid=True, semantic_valid=True, error=None, started=started)
            )
            metrics = metrics.finalize(succeeded=True, final_error_code=None)
            return StageResult(
                stage_name=self.stage_name,
                status=StageStatus.SUCCESS,
                output=contract,
                validation=semantic_result,  # may still carry warnings
                error=None,
                metrics=metrics,
                confidence=self.collect_confidence(contract),
                attempts_used=attempt,
            )

    # ------------------------------------------------------------ step helpers
    def _check_tool_call(self, response: LLMToolResponse, attempt: int) -> StageError | None:
        """Return a :class:`StageError` if no usable tool call was produced, else ``None``."""

        if response.tool_input is None or response.tool_name != self.tool_name:
            # Distinguish a truncated call from a plain no-call for clearer metrics.
            if response.stop_reason == "max_tokens":
                return StageError(
                    code=ErrorCode.TRUNCATED_OUTPUT,
                    message="model output was truncated before a complete tool call",
                    attempt=attempt,
                    recoverable=True,
                    details={"stop_reason": response.stop_reason},
                )
            return StageError(
                code=ErrorCode.TOOL_NOT_CALLED,
                message=f"model did not call the required tool {self.tool_name!r}",
                attempt=attempt,
                recoverable=True,
                details={"stop_reason": response.stop_reason, "tool_name": response.tool_name},
            )
        return None

    def _schema_validate(self, tool_input: dict) -> tuple[TContract | None, ValidationResult]:
        """Validate raw tool input against the contract model."""

        try:
            contract = self.contract_model().model_validate(tool_input)
        except ValidationError as exc:
            return None, ValidationResult.from_pydantic_error(exc)
        return contract, ValidationResult.success()

    def _semantic_validate(self, contract: TContract, context: TContext) -> ValidationResult:
        """Run every stage-specific validator and merge their results."""

        result = ValidationResult.success()
        for validator in self.semantic_validators():
            result = result.merge(validator.validate(contract, context))
        return result

    def _tool_call_correction(self, error: StageError) -> LLMMessage:
        """Deterministic correction message for a missing/malformed tool call."""

        return LLMMessage(
            role="user",
            content=(
                f"You must respond by calling the {self.tool_name!r} tool with valid input. "
                f"Previous attempt failed: {error.message}. Call the tool now."
            ),
        )

    def _attempt_metrics(
        self,
        attempt: int,
        response: LLMToolResponse | None,
        *,
        schema_valid: bool,
        semantic_valid: bool,
        error: StageError | None,
        started: float | None,
    ) -> AttemptMetrics:
        """Build :class:`AttemptMetrics` for one attempt (response may be ``None`` on client error)."""

        latency_ms: float | None = None
        if started is not None and self._clock is not None:
            latency_ms = max(0.0, (self._clock() - started) * 1000.0)
        return AttemptMetrics(
            attempt=attempt,
            model_id=response.model_id if response else "unknown",
            input_tokens=response.input_tokens if response else 0,
            output_tokens=response.output_tokens if response else 0,
            stop_reason=response.stop_reason if response else "client_error",
            schema_valid=schema_valid,
            semantic_valid=semantic_valid,
            error_code=error.code if error else None,
            latency_ms=latency_ms,
        )

    def _wait(self, delay_seconds: float) -> None:
        """Sleep before a retry, only if a positive delay is configured."""

        if delay_seconds > 0:
            self._sleep(delay_seconds)

    def _fail(
        self,
        metrics: StageMetrics,
        error: StageError,
        validation: ValidationResult,
        attempt: int,
        *,
        exhausted: bool,
    ) -> StageResult[TContract]:
        """Assemble a terminal failure :class:`StageResult`."""

        status = StageStatus.FAILED_EXHAUSTED if exhausted else StageStatus.for_error(error.code)
        metrics = metrics.finalize(succeeded=False, final_error_code=error.code)
        return StageResult(
            stage_name=self.stage_name,
            status=status,
            output=None,
            validation=validation,
            error=error,
            metrics=metrics,
            confidence=None,
            attempts_used=attempt,
        )
