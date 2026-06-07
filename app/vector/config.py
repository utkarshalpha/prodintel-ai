"""Vector infrastructure configuration (Phase 6C-3A).

Holds the choice of vector backend (the deterministic in-memory fake vs ChromaDB),
the Chroma connection details, and the embedding model/dimensionality the corpus is
pinned to. Mirrors :class:`app.ai_clients.config.ClaudeClientConfig`: a frozen
Pydantic model with a :meth:`VectorConfig.from_env` builder that raises a typed
:class:`ChromaConfigurationError` for bad or missing settings.

Imports no ``chromadb`` -- this module only describes configuration; the adapters and
wiring (later 6C-3 sub-steps) consume it.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.vector.errors import ChromaConfigurationError
from app.vector.naming import build_collection_name

__all__ = ["VectorBackend", "ChromaMode", "VectorConfig"]

DEFAULT_COLLECTION_PREFIX = "prodintel_knowledge"


class VectorBackend(str, Enum):
    """Which vector implementation to wire."""

    MEMORY = "memory"  # deterministic in-memory fake (default; offline)
    CHROMA = "chroma"  # real ChromaDB


class ChromaMode(str, Enum):
    """How to connect to ChromaDB (only relevant when backend is ``chroma``)."""

    EPHEMERAL = "ephemeral"  # in-process, non-persistent
    PERSISTENT = "persistent"  # on-disk at persist_directory
    HTTP = "http"  # remote server at host:port


class VectorConfig(BaseModel):
    """Immutable vector-infrastructure configuration."""

    model_config = ConfigDict(frozen=True)

    backend: VectorBackend = Field(default=VectorBackend.MEMORY)
    mode: ChromaMode = Field(default=ChromaMode.EPHEMERAL, description="Chroma connection mode.")
    persist_directory: str | None = Field(default=None, description="On-disk dir for persistent mode.")
    host: str | None = Field(default=None, description="Chroma server host for http mode.")
    port: int | None = Field(default=None, description="Chroma server port for http mode.")
    collection_prefix: str = Field(default=DEFAULT_COLLECTION_PREFIX, min_length=1)
    embedding_model_id: str | None = Field(default=None, description="Embedding model id (required for chroma).")
    dim: int | None = Field(default=None, description="Embedding dimensionality (required for chroma).")
    distance: Literal["cosine"] = Field(default="cosine", description="Similarity space (cosine only).")

    @model_validator(mode="after")
    def _check_consistency(self) -> "VectorConfig":
        """Enforce the fields each backend/mode requires (raises ChromaConfigurationError)."""

        if self.backend is VectorBackend.CHROMA:
            if not self.embedding_model_id:
                raise ChromaConfigurationError("EMBEDDING_MODEL_ID is required when VECTOR_BACKEND=chroma")
            if self.dim is None:
                raise ChromaConfigurationError("EMBEDDING_DIM is required when VECTOR_BACKEND=chroma")
            if self.dim < 1:
                raise ChromaConfigurationError("EMBEDDING_DIM must be >= 1")
            if self.mode is ChromaMode.PERSISTENT and not self.persist_directory:
                raise ChromaConfigurationError("CHROMA_PERSIST_DIR is required when CHROMA_MODE=persistent")
            if self.mode is ChromaMode.HTTP and (not self.host or self.port is None):
                raise ChromaConfigurationError("CHROMA_HOST and CHROMA_PORT are required when CHROMA_MODE=http")
        return self

    def collection_name(self) -> str:
        """The deterministic Chroma collection name for this config.

        Only defined for the chroma backend (embedding_model_id/dim are guaranteed
        present there by validation).
        """

        if self.backend is not VectorBackend.CHROMA:
            raise ChromaConfigurationError("collection_name is only defined for the chroma backend")
        return build_collection_name(self.collection_prefix, self.embedding_model_id, self.dim)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "VectorConfig":
        """Build a config from environment variables.

        Reads ``VECTOR_BACKEND`` (default ``memory``), ``CHROMA_MODE`` (default
        ``ephemeral``), ``CHROMA_PERSIST_DIR``, ``CHROMA_HOST``, ``CHROMA_PORT``,
        ``CHROMA_COLLECTION_PREFIX``, ``EMBEDDING_MODEL_ID``, and ``EMBEDDING_DIM``.
        Raises :class:`ChromaConfigurationError` for unknown enum values, non-integer
        numeric fields, or a missing required field.
        """

        source = os.environ if env is None else env
        backend = _parse_enum(source, "VECTOR_BACKEND", VectorBackend, VectorBackend.MEMORY)
        mode = _parse_enum(source, "CHROMA_MODE", ChromaMode, ChromaMode.EPHEMERAL)

        return cls(
            backend=backend,
            mode=mode,
            persist_directory=source.get("CHROMA_PERSIST_DIR") or None,
            host=source.get("CHROMA_HOST") or None,
            port=_parse_int(source, "CHROMA_PORT"),
            collection_prefix=source.get("CHROMA_COLLECTION_PREFIX") or DEFAULT_COLLECTION_PREFIX,
            embedding_model_id=source.get("EMBEDDING_MODEL_ID") or None,
            dim=_parse_int(source, "EMBEDDING_DIM"),
        )


def _parse_enum(source: Mapping[str, str], key: str, enum_cls, default):
    raw = source.get(key)
    if not raw:
        return default
    try:
        return enum_cls(raw)
    except ValueError:
        allowed = [member.value for member in enum_cls]
        raise ChromaConfigurationError(f"{key} must be one of {allowed}, got {raw!r}") from None


def _parse_int(source: Mapping[str, str], key: str) -> int | None:
    raw = source.get(key)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        raise ChromaConfigurationError(f"{key} must be an integer, got {raw!r}") from None
