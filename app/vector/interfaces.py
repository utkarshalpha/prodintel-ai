"""Provider-neutral interfaces and DTOs for the vector infrastructure (Phase 6C).

The same single-seam design as ``app.ai_runtime.interfaces``: the rest of the system
talks only to two ``@runtime_checkable`` Protocols -- :class:`EmbeddingClient` and
:class:`VectorIndex` -- whose request/response shapes are immutable Pydantic DTOs.
A real ChromaDB adapter (6C-3) and a deterministic in-memory fake both implement
them, so every ingestion/retrieval path runs with no network and no vector store.

Conventions
-----------
* DTOs are frozen (``ConfigDict(frozen=True)``), mirroring the harness DTOs.
* ``EmbeddingVector.values`` is a ``tuple`` so a vector is hashable and immutable.
* The index uses **dot-product** similarity, which equals cosine similarity because
  embeddings are expected to be **L2-normalized** (the fake embedder guarantees it,
  and production embedders are configured to normalize).
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "EmbeddingVector",
    "VectorRecord",
    "VectorMatch",
    "VectorFilter",
    "EmbeddingClient",
    "VectorIndex",
]

# Scalar value types permitted in vector metadata (flat scalars only -- never text).
_Scalar = str | int | float | bool


class EmbeddingVector(BaseModel):
    """An embedding: its values, its dimensionality, and the model that produced it."""

    model_config = ConfigDict(frozen=True)

    values: tuple[float, ...] = Field(..., description="The embedding components.")
    dim: int = Field(..., ge=1, description="Dimensionality; must equal len(values).")
    model_id: str = Field(..., min_length=1, description="Identifier of the producing embedding model.")

    @model_validator(mode="after")
    def _check_length(self) -> "EmbeddingVector":
        if len(self.values) != self.dim:
            raise ValueError(f"embedding has {len(self.values)} values but dim={self.dim}")
        return self


class VectorRecord(BaseModel):
    """One upsertable record: a chunk's id, its vector, and flat scalar metadata.

    ``chunk_id`` is ``str(KnowledgeChunk.id)`` -- the deterministic vector key that
    makes ingestion resume orphan-proof (re-upsert overwrites the same id). Metadata
    holds only flat scalars (e.g. ``corpus_version``, ``framework``, ``ordinal``) and
    **never the chunk text** (Postgres is the system of record for text).
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(..., min_length=1)
    vector: EmbeddingVector = Field(...)
    metadata: dict[str, _Scalar] = Field(default_factory=dict)


class VectorMatch(BaseModel):
    """A single query hit: the matched chunk id and its similarity score."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(..., min_length=1)
    score: float = Field(...)


class VectorFilter(BaseModel):
    """Exact-match metadata filter (AND semantics over ``equals``)."""

    model_config = ConfigDict(frozen=True)

    equals: dict[str, _Scalar] = Field(default_factory=dict)


@runtime_checkable
class EmbeddingClient(Protocol):
    """The single boundary between the system and an embedding provider."""

    @property
    def model_id(self) -> str:
        """Identifier of the embedding model (stamped onto persisted rows)."""
        ...

    @property
    def dim(self) -> int:
        """Dimensionality of the vectors this client produces."""
        ...

    def embed(self, texts: Sequence[str]) -> Sequence[EmbeddingVector]:
        """Embed a batch of texts. Output length and order match the input.

        Must raise :class:`~app.vector.errors.EmbeddingClientError` on provider
        failure.
        """
        ...

    def embed_one(self, text: str) -> EmbeddingVector:
        """Embed a single text."""
        ...


@runtime_checkable
class VectorIndex(Protocol):
    """The single boundary between the system and a vector store."""

    def upsert(self, records: Sequence[VectorRecord]) -> None:
        """Insert or overwrite records, keyed by ``chunk_id`` (idempotent)."""
        ...

    def query(
        self,
        vector: EmbeddingVector,
        *,
        top_k: int,
        where: VectorFilter | None = None,
    ) -> Sequence[VectorMatch]:
        """Return up to ``top_k`` nearest records, optionally metadata-filtered.

        Deterministically ordered by ``(score desc, chunk_id asc)``. Defined for the
        future retrieval phase; no wired path calls it in Phase 6C.
        """
        ...

    def delete(self, chunk_ids: Sequence[str]) -> None:
        """Remove records by id. Missing ids are ignored."""
        ...
