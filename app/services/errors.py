"""Service-layer errors, mapped to HTTP status codes by the API layer."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid a runtime import cycle; only needed for typing
    from app.ai_contracts.stage1_signal import ParsedSignalContract
    from app.ai_runtime.stage_result import StageResult

__all__ = [
    "SignalServiceError",
    "SignalNotFoundError",
    "AnalysisNotFoundError",
    "AnalysisFailedError",
    "FeatureNotFoundError",
    "SignalsNotAnalyzedError",
    "FeatureExtractionFailedError",
    "ConflictNotFoundError",
    "ConflictDetectionFailedError",
    "DecisionNotFoundError",
    "DecisionSynthesisFailedError",
    "KnowledgeSourceNotFoundError",
    "CorpusEmbeddingModelMismatchError",
    "ChunkOrdinalError",
    "KnowledgeIngestionFailedError",
    "RetrievalFailedError",
]


class SignalServiceError(Exception):
    """Base class for all signal-service errors."""


class SignalNotFoundError(SignalServiceError):
    """The requested signal does not exist (maps to HTTP 404)."""

    def __init__(self, signal_id: uuid.UUID) -> None:
        self.signal_id = signal_id
        super().__init__(f"signal {signal_id} not found")


class AnalysisNotFoundError(SignalServiceError):
    """The signal exists but has not been analyzed yet (maps to HTTP 404)."""

    def __init__(self, signal_id: uuid.UUID) -> None:
        self.signal_id = signal_id
        super().__init__(f"no analysis found for signal {signal_id}")


class AnalysisFailedError(SignalServiceError):
    """Stage 1 ran but produced no valid, grounded result (maps to HTTP 422).

    Carries the terminal :class:`StageResult` so the API can surface a precise,
    structured reason (status, attempts, error code) -- "insufficient evidence to
    analyze" is a legitimate product outcome, not a hidden failure.
    """

    def __init__(self, signal_id: uuid.UUID, stage_result: "StageResult[ParsedSignalContract]") -> None:
        self.signal_id = signal_id
        self.stage_result = stage_result
        reason = stage_result.error.message if stage_result.error else "unknown"
        super().__init__(f"analysis failed for signal {signal_id}: {reason}")


class FeatureNotFoundError(SignalServiceError):
    """The requested feature does not exist (maps to HTTP 404)."""

    def __init__(self, feature_id: uuid.UUID) -> None:
        self.feature_id = feature_id
        super().__init__(f"feature {feature_id} not found")


class SignalsNotAnalyzedError(SignalServiceError):
    """Feature extraction requires analyzed signals, but some are not (HTTP 422)."""

    def __init__(self, signal_ids: list[uuid.UUID]) -> None:
        self.signal_ids = signal_ids
        joined = ", ".join(str(s) for s in signal_ids)
        super().__init__(f"these signals must be analyzed before extraction: {joined}")


class FeatureExtractionFailedError(SignalServiceError):
    """Stage 2 ran but produced no valid, fully-attributed feature set (HTTP 422)."""

    def __init__(self, stage_result: "StageResult") -> None:  # noqa: F821 - runtime-only
        self.stage_result = stage_result
        reason = stage_result.error.message if stage_result.error else "unknown"
        super().__init__(f"feature extraction failed: {reason}")


class ConflictNotFoundError(SignalServiceError):
    """The requested conflict does not exist (maps to HTTP 404)."""

    def __init__(self, conflict_id: uuid.UUID) -> None:
        self.conflict_id = conflict_id
        super().__init__(f"conflict {conflict_id} not found")


class ConflictDetectionFailedError(SignalServiceError):
    """Stage 3 ran but produced no valid, evidence-traceable result (HTTP 422)."""

    def __init__(self, stage_result: "StageResult") -> None:  # noqa: F821 - runtime-only
        self.stage_result = stage_result
        reason = stage_result.error.message if stage_result.error else "unknown"
        super().__init__(f"conflict detection failed: {reason}")


class DecisionNotFoundError(SignalServiceError):
    """The requested decision does not exist (maps to HTTP 404)."""

    def __init__(self, decision_id: uuid.UUID) -> None:
        self.decision_id = decision_id
        super().__init__(f"decision {decision_id} not found")


class DecisionSynthesisFailedError(SignalServiceError):
    """Stage 4 ran but produced no valid, evidence-traceable result (HTTP 422)."""

    def __init__(self, stage_result: "StageResult") -> None:  # noqa: F821 - runtime-only
        self.stage_result = stage_result
        reason = stage_result.error.message if stage_result.error else "unknown"
        super().__init__(f"decision synthesis failed: {reason}")


class KnowledgeSourceNotFoundError(SignalServiceError):
    """The requested knowledge source does not exist (maps to HTTP 404)."""

    def __init__(self, source_id: uuid.UUID) -> None:
        self.source_id = source_id
        super().__init__(f"knowledge source {source_id} not found")


class CorpusEmbeddingModelMismatchError(SignalServiceError):
    """The embedding model does not agree with the corpus_version's pinned model.

    Raised when (a) the contract's declared ``embedding_model_id`` does not match the
    wired embedding client, or (b) a ``corpus_version`` already ingested under model A
    is being ingested with model B. A corpus snapshot is single-model by invariant;
    switching models requires a new ``corpus_version``. Maps to HTTP 409.
    """

    def __init__(self, corpus_version: str, expected: str, actual: str) -> None:
        self.corpus_version = corpus_version
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"embedding model mismatch for corpus_version {corpus_version!r}: "
            f"expected {expected!r}, got {actual!r}"
        )


class ChunkOrdinalError(SignalServiceError):
    """The source's chunk ordinals are not a contiguous 0-based sequence (HTTP 422)."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class KnowledgeIngestionFailedError(SignalServiceError):
    """Embedding or vector-index upsert failed during ingestion (maps to HTTP 502).

    The system-of-record rows are committed (``chroma_id`` NULL), so the operation is
    resumable via a re-``ingest`` or ``reembed_pending``. Carries the underlying
    vector-layer error as ``__cause__`` for a precise reason.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"knowledge ingestion failed: {reason}")


class RetrievalFailedError(SignalServiceError):
    """Query embedding or vector query failed during retrieval (maps to HTTP 502).

    Retrieval is read-only, so nothing is persisted; the operation can simply be
    retried. Carries the underlying vector-layer error as ``__cause__``.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"retrieval failed: {reason}")
