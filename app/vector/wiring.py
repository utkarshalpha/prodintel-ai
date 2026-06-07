"""Vector infrastructure wiring (Phase 6C-3C).

Turns a :class:`~app.vector.config.VectorConfig` into a concrete
``(EmbeddingClient, VectorIndex)`` pair behind the Protocols, validates a real Chroma
backend at startup, and registers the pair on ``app.state`` -- mirroring
:func:`app.ai_clients.wiring.attach_claude_runner` (which sets ``app.state.stageN_runner``).

Design properties:

* **Lazy:** importing this module pulls in no ``chromadb``. The adapter modules are
  themselves chromadb-free at import; ``chromadb`` loads only inside the adapters'
  ``from_config`` when the chroma backend is wired with a real client.
* **Offline-first:** the ``memory`` backend (the default) wires the deterministic
  ``HashEmbeddingClient`` + ``InMemoryVectorIndex`` with no external dependency.
* **Fail-fast, not fallback:** an explicitly-enabled-but-broken Chroma backend raises
  :class:`ChromaConfigurationError` at startup; it never silently degrades to fakes.
* **Framework-neutral:** ``app`` is duck-typed (anything with a ``.state``) so the
  vector layer imports no web framework.

DI boundary: the seam is ``app.state.embedding_client`` / ``app.state.vector_index``.
FastAPI ``get_*`` providers, an ingest/retrieval endpoint, and the
``create_production_app`` startup hookup are deferred to the retrieval phase.
"""

from __future__ import annotations

from typing import Any

from app.vector.config import VectorBackend, VectorConfig
from app.vector.embedding_chroma import ChromaEmbeddingAdapter
from app.vector.embedding_fake import HashEmbeddingClient
from app.vector.errors import ChromaConfigurationError, VectorIndexError
from app.vector.index_chroma import ChromaVectorIndex
from app.vector.index_fake import InMemoryVectorIndex
from app.vector.interfaces import EmbeddingClient, EmbeddingVector, VectorIndex

__all__ = [
    "DEFAULT_MEMORY_DIM",
    "make_embedding_client",
    "make_vector_index",
    "validate_vector_infra",
    "attach_vector_infra",
]

#: Dimensionality used by the in-memory backend when the config does not pin one.
DEFAULT_MEMORY_DIM = 16
_DEFAULT_MEMORY_MODEL_ID = "hash-fake-v1"


def _memory_dim(config: VectorConfig) -> int:
    return config.dim if config.dim is not None else DEFAULT_MEMORY_DIM


def make_embedding_client(
    config: VectorConfig, *, embedding_function: Any | None = None
) -> EmbeddingClient:
    """Build the embedding client for ``config.backend``.

    ``embedding_function`` is a test/advanced injection for the chroma backend.
    """

    if config.backend is VectorBackend.CHROMA:
        return ChromaEmbeddingAdapter.from_config(config, embedding_function=embedding_function)
    return HashEmbeddingClient(
        dim=_memory_dim(config),
        model_id=config.embedding_model_id or _DEFAULT_MEMORY_MODEL_ID,
    )


def make_vector_index(config: VectorConfig, *, client: Any | None = None) -> VectorIndex:
    """Build the vector index for ``config.backend``.

    ``client`` is a test/advanced injection for the chroma backend.
    """

    if config.backend is VectorBackend.CHROMA:
        return ChromaVectorIndex.from_config(config, client=client)
    return InMemoryVectorIndex(dim=_memory_dim(config))


def validate_vector_infra(
    embedding_client: EmbeddingClient, vector_index: VectorIndex, config: VectorConfig
) -> None:
    """Startup validation: dim consistency, and (chroma) a read-connectivity probe.

    Raises :class:`ChromaConfigurationError` on any inconsistency or unreachable store.
    """

    if embedding_client.dim != vector_index.dim:
        raise ChromaConfigurationError(
            f"embedding dim {embedding_client.dim} != index dim {vector_index.dim}"
        )
    if config.dim is not None and embedding_client.dim != config.dim:
        raise ChromaConfigurationError(
            f"embedding dim {embedding_client.dim} != configured dim {config.dim}"
        )

    if config.backend is VectorBackend.CHROMA:
        probe = EmbeddingVector(
            values=_unit_values(vector_index.dim),
            dim=vector_index.dim,
            model_id=embedding_client.model_id,
        )
        try:
            vector_index.query(probe, top_k=1)  # read probe against the live collection
        except VectorIndexError as exc:
            raise ChromaConfigurationError(f"vector store read probe failed: {exc}") from exc


def attach_vector_infra(
    app: Any,
    config: VectorConfig | None = None,
    *,
    client: Any | None = None,
    embedding_function: Any | None = None,
) -> tuple[EmbeddingClient, VectorIndex]:
    """Build, validate, and register the vector pair on ``app.state``.

    Resolves ``config`` from the environment if not given. Returns the pair for
    convenience. Raises :class:`ChromaConfigurationError` on a misconfigured chroma
    backend (the memory backend never fails).
    """

    cfg = config if config is not None else VectorConfig.from_env()
    embedding_client = make_embedding_client(cfg, embedding_function=embedding_function)
    vector_index = make_vector_index(cfg, client=client)
    validate_vector_infra(embedding_client, vector_index, cfg)
    app.state.embedding_client = embedding_client
    app.state.vector_index = vector_index
    return embedding_client, vector_index


def _unit_values(dim: int) -> tuple[float, ...]:
    return tuple(1.0 if i == 0 else 0.0 for i in range(dim))
