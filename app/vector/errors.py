"""Errors for the vector infrastructure layer (Phase 6C).

Mirrors the harness's split between retryable transport failures and fatal config
errors (see ``app.ai_runtime.errors``). All inherit from :class:`VectorError` so a
caller can catch the whole layer with one type; the ingestion service (6C-2) maps
them to its own service-level failures.
"""

from __future__ import annotations

__all__ = [
    "VectorError",
    "EmbeddingClientError",
    "VectorIndexError",
    "EmbeddingDimensionMismatchError",
    "ChromaConfigurationError",
]


class VectorError(Exception):
    """Base class for every error raised by the vector layer."""


class EmbeddingClientError(VectorError):
    """Transport/API failure while computing embeddings (retryable).

    The analog of ``LLMClientError``: a transient failure of the embedding provider.
    The ingestion flow leaves the system-of-record rows committed (``chroma_id``
    NULL), so the operation is resumable.
    """


class VectorIndexError(VectorError):
    """Failure of a vector-index operation (upsert/query/delete)."""


class EmbeddingDimensionMismatchError(VectorError):
    """An embedding's dimensionality does not match what the index expects.

    A fatal configuration error -- raised when a vector of dimension D is presented
    to an index configured for a different dimension (or vice versa). Never resolved
    by retrying.
    """


class ChromaConfigurationError(VectorError):
    """The vector configuration is invalid or incomplete.

    Raised by :meth:`app.vector.config.VectorConfig.from_env` (and the config's own
    cross-field validation) for unknown enum values, non-integer numeric fields, or a
    required field missing for the selected backend/mode. A fatal setup error.
    """
