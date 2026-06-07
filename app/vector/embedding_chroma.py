"""ChromaDB-backed :class:`~app.vector.interfaces.EmbeddingClient` adapter (6C-3B).

Wraps an embedding function -- by default a Chroma ``EmbeddingFunction`` (a local
sentence-transformers model needing no API key) -- and exposes it through the
provider-neutral :class:`EmbeddingClient` Protocol. Output vectors are
**L2-normalized** so the index's dot-product equals cosine similarity, matching the
in-memory fake.

``chromadb`` is optional and imported **lazily** inside :meth:`from_config`; importing
this module pulls in no ``chromadb``. Provider failures are wrapped into
:class:`EmbeddingClientError`; a vector whose dimensionality disagrees with the
configured ``dim`` raises :class:`EmbeddingDimensionMismatchError` (a fatal config
error). The embedding function is injectable, so the adapter is fully testable
offline with a pure-Python double.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from app.vector.config import VectorBackend, VectorConfig
from app.vector.errors import (
    ChromaConfigurationError,
    EmbeddingClientError,
    EmbeddingDimensionMismatchError,
)
from app.vector.interfaces import EmbeddingVector

__all__ = ["ChromaEmbeddingAdapter"]


class ChromaEmbeddingAdapter:
    """An :class:`EmbeddingClient` backed by a (Chroma) embedding function."""

    def __init__(self, embedding_function: Any, *, model_id: str, dim: int) -> None:
        if dim < 1:
            raise ValueError("dim must be >= 1")
        if not model_id:
            raise ValueError("model_id must be non-empty")
        self._ef = embedding_function
        self._model_id = model_id
        self._dim = dim

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        inputs = list(texts)
        if not inputs:
            return []
        try:
            raw = list(self._ef(inputs))
        except Exception as exc:  # noqa: BLE001 - no raw provider error may escape
            raise EmbeddingClientError(f"embedding provider failed: {exc}") from exc

        if len(raw) != len(inputs):
            raise EmbeddingClientError(
                f"embedding provider returned {len(raw)} vectors for {len(inputs)} inputs"
            )

        vectors: list[EmbeddingVector] = []
        for vec in raw:
            values = [float(component) for component in vec]
            if len(values) != self._dim:
                raise EmbeddingDimensionMismatchError(
                    f"embedding provider returned dim={len(values)}, expected {self._dim}"
                )
            vectors.append(
                EmbeddingVector(values=_l2_normalize(values), dim=self._dim, model_id=self._model_id)
            )
        return vectors

    def embed_one(self, text: str) -> EmbeddingVector:
        return self.embed([text])[0]

    @classmethod
    def from_config(
        cls, config: VectorConfig, *, embedding_function: Any | None = None
    ) -> "ChromaEmbeddingAdapter":
        """Build from a chroma-backed :class:`VectorConfig` (lazily importing chromadb).

        ``embedding_function`` may be injected (tests); otherwise the default local
        sentence-transformers function for ``config.embedding_model_id`` is created.
        """

        if config.backend is not VectorBackend.CHROMA:
            raise ChromaConfigurationError("ChromaEmbeddingAdapter requires backend=chroma")
        function = embedding_function
        if function is None:
            try:
                from chromadb.utils import embedding_functions  # noqa: PLC0415 - lazy optional dep
            except ImportError as exc:
                raise ChromaConfigurationError(
                    "chromadb is not installed; install the 'vector' optional dependency"
                ) from exc
            function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=config.embedding_model_id
            )
        return cls(function, model_id=config.embedding_model_id, dim=config.dim)


def _l2_normalize(values: list[float]) -> tuple[float, ...]:
    norm = math.sqrt(sum(component * component for component in values))
    if norm == 0.0:
        return tuple(1.0 if i == 0 else 0.0 for i in range(len(values)))
    return tuple(component / norm for component in values)
