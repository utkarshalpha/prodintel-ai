"""SignalService -- the application service for Stage 1 (PART 1 + PART 5).

Orchestrates the use cases the API exposes, owning the transaction boundary and the
dedupe/analysis policy while delegating persistence to repositories and analysis to
the injected :class:`Stage1SignalRunner`. It contains no SQL and no LLM calls of its
own, which keeps it unit-testable with fakes or an in-memory database.

Dependencies (all injected)
---------------------------
* ``session`` -- the unit-of-work the service commits/rolls back.
* ``signal_repo`` / ``parsed_repo`` -- persistence.
* ``runner`` -- the Stage 1 runner (which wraps the Claude client; the client itself
  is out of scope for this layer).

Use cases
---------
* :meth:`create_signal` -- hash + dedupe + persist an immutable signal (idempotent).
* :meth:`get_signal` -- fetch a signal or raise :class:`SignalNotFoundError`.
* :meth:`analyze_signal` -- run Stage 1 and persist only a grounded result.
* :meth:`get_analysis` -- fetch the persisted analysis or raise.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.ai_contracts.enums import StakeholderType
from app.models.signal import ParsedSignal, Signal
from app.repositories.parsed_signal_repository import ParsedSignalRepository
from app.repositories.signal_repository import SignalRepository
from app.observability.logging import get_logger, log_context
from app.services.errors import (
    AnalysisFailedError,
    AnalysisNotFoundError,
    SignalNotFoundError,
)
from app.stages.stage1.runner import Stage1Context, Stage1SignalRunner

__all__ = ["SignalService", "SignalCreateResult", "SignalAnalysisResult"]

_logger = get_logger(__name__)
_STAGE = "stage1"


@dataclass(frozen=True)
class SignalCreateResult:
    """Outcome of a create call. ``created`` is ``False`` when an identical signal
    already existed (idempotent create), so the API can return 200 vs 201."""

    signal: Signal
    created: bool


@dataclass(frozen=True)
class SignalAnalysisResult:
    """Outcome of an analyze call: the persisted analysis plus the run's
    :class:`StageResult` (metrics, confidence, attempts) for the response."""

    parsed_signal: ParsedSignal
    stage_result: object  # StageResult[ParsedSignalContract]; kept loose to avoid import weight


class SignalService:
    """Application service coordinating signal creation, retrieval, and analysis."""

    def __init__(
        self,
        session: Session,
        signal_repo: SignalRepository,
        parsed_repo: ParsedSignalRepository,
        runner: Stage1SignalRunner,
    ) -> None:
        self._session = session
        self._signals = signal_repo
        self._parsed = parsed_repo
        self._runner = runner

    # ------------------------------------------------------------------ create
    def create_signal(
        self,
        *,
        source_type: StakeholderType,
        raw_text: str,
        source_ref: str | None = None,
        workspace_id: uuid.UUID | None = None,
    ) -> SignalCreateResult:
        """Persist an immutable signal, deduplicating by content hash.

        If an identical signal already exists, returns it with ``created=False``
        and writes nothing (idempotent). Otherwise inserts and commits.
        """

        content_hash = self._content_hash(raw_text)
        existing = self._signals.get_by_content_hash(content_hash)
        if existing is not None:
            _logger.info(
                "signal_deduplicated",
                extra={"event": "signal_deduplicated", "signal_id": str(existing.id)},
            )
            return SignalCreateResult(signal=existing, created=False)

        signal = Signal(
            workspace_id=workspace_id,
            source_type=source_type,
            raw_text=raw_text,
            source_ref=source_ref,
            content_hash=content_hash,
        )
        self._signals.add(signal)
        self._session.commit()
        _logger.info(
            "signal_created",
            extra={
                "event": "db_write",
                "entity": "signal",
                "signal_id": str(signal.id),
                "source_type": source_type.value,
            },
        )
        return SignalCreateResult(signal=signal, created=True)

    # --------------------------------------------------------------------- get
    def get_signal(self, signal_id: uuid.UUID) -> Signal:
        """Return a signal or raise :class:`SignalNotFoundError`."""

        signal = self._signals.get(signal_id)
        if signal is None:
            raise SignalNotFoundError(signal_id)
        return signal

    # ----------------------------------------------------------------- analyze
    def analyze_signal(self, signal_id: uuid.UUID) -> SignalAnalysisResult:
        """Run Stage 1 over a signal and persist a grounded analysis.

        Structured in three phases so that **no database transaction is held during
        the Claude call**: the read transaction is committed (releasing the
        connection) before the AI phase, and writes happen in a fresh transaction.

        Raises :class:`SignalNotFoundError` if the signal is missing and
        :class:`AnalysisFailedError` if the stage did not produce a valid, grounded
        result (nothing is persisted in that case).
        """

        # ---- Phase 1: READ -- fetch inputs, then release the connection. --------
        signal = self.get_signal(signal_id)
        raw_text = signal.raw_text
        source_type = signal.source_type
        self._session.commit()  # end the read transaction before the AI call

        # ---- Phase 2: AI EXECUTION -- no DB transaction/connection held. --------
        with log_context(signal_id=str(signal_id)):
            _logger.info("stage_started", extra={"event": "stage_started", "stage": _STAGE, "signal_id": str(signal_id)})
            context = Stage1Context(signal_id=signal_id, raw_text=raw_text, source_type=source_type)
            result = self._runner.run(context)
            self._log_stage_attempts(result)

            if not result.succeeded or result.output is None:
                _logger.warning(
                    "stage_failed",
                    extra={
                        "event": "stage_failed",
                        "stage": _STAGE,
                        "signal_id": str(signal_id),
                        "status": result.status.value,
                        "error_code": result.error.code.value if result.error else None,
                        "attempts": result.attempts_used,
                    },
                )
                raise AnalysisFailedError(signal_id, result)

            # ---- Phase 3: WRITE -- fresh transaction. ---------------------------
            parsed = ParsedSignal.from_contract(result.output)
            self._parsed.upsert_for_signal(parsed)
            self._session.commit()
            _logger.info(
                "db_write",
                extra={"event": "db_write", "entity": "parsed_signal", "signal_id": str(signal_id)},
            )
            _logger.info(
                "stage_succeeded",
                extra={
                    "event": "stage_succeeded",
                    "stage": _STAGE,
                    "signal_id": str(signal_id),
                    "attempts": result.attempts_used,
                    "input_tokens": result.metrics.total_input_tokens,
                    "output_tokens": result.metrics.total_output_tokens,
                },
            )
        return SignalAnalysisResult(parsed_signal=parsed, stage_result=result)

    @staticmethod
    def _log_stage_attempts(result: object) -> None:
        """Emit one structured log per failed attempt (validation/retry events)."""

        for attempt in result.metrics.attempts:  # type: ignore[attr-defined]
            if attempt.error_code is not None:
                _logger.warning(
                    "stage_retry",
                    extra={
                        "event": "retry_event",
                        "stage": _STAGE,
                        "attempt": attempt.attempt,
                        "error_code": attempt.error_code.value,
                    },
                )

    # ------------------------------------------------------------ get analysis
    def get_analysis(self, signal_id: uuid.UUID) -> ParsedSignal:
        """Return the persisted analysis or raise :class:`AnalysisNotFoundError`."""

        parsed = self._parsed.get_by_signal_id(signal_id)
        if parsed is None:
            raise AnalysisNotFoundError(signal_id)
        return parsed

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _content_hash(raw_text: str) -> str:
        """Stable SHA-256 of the signal text, used for deduplication.

        Hashes the exact stored text so identical submissions collide; this is the
        text the model also receives, so character offsets stay aligned.
        """

        return hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
