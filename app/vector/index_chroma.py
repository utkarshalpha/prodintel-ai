"""ChromaDB-backed :class:`~app.vector.interfaces.VectorIndex` adapter (Phase 6C-3B).

Chroma is used as a **pure vector store**: the service computes embeddings and hands
explicit vectors to :meth:`ChromaVectorIndex.upsert`, keyed by ``str(chunk.id)``.
Chunk text is never stored here (Postgres is the system of record); only
``ids + embeddings + metadata`` go to Chroma. The collection is created with cosine
space so dot-product-on-normalized-vectors parity with the in-memory fake holds.

``chromadb`` is an **optional** dependency, imported **lazily** inside
:meth:`from_config` / :func:`_build_chroma_client`. Importing this module pulls in no
``chromadb``. Every provider call is wrapped so no raw ``chromadb`` exception escapes:
index operations raise :class:`VectorIndexError`; a dimension mismatch raises
:class:`EmbeddingDimensionMismatchError` (a fatal config error, not wrapped).

The adapter accepts an injectable ``collection`` so it is fully testable offline with
a pure-Python double.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.vector.config import ChromaMode, VectorBackend, VectorConfig
from app.vector.errors import (
    ChromaConfigurationError,
    EmbeddingDimensionMismatchError,
    VectorIndexError,
)
from app.vector.interfaces import EmbeddingVector, VectorFilter, VectorMatch, VectorRecord

__all__ = ["ChromaVectorIndex"]


class ChromaVectorIndex:
    """A :class:`VectorIndex` backed by a ChromaDB collection."""

    def __init__(self, collection: Any, *, dim: int, model_id: str = "chroma") -> None:
        self._collection = collection
        self._dim = dim
        self._model_id = model_id  # used only to reconstruct vectors in snapshot()

    @property
    def dim(self) -> int:
        return self._dim

    def upsert(self, records: Sequence[VectorRecord]) -> None:
        items = list(records)
        if not items:
            return
        for record in items:
            self._require_dim(record.vector)
        ids = [record.chunk_id for record in items]
        embeddings = [list(record.vector.values) for record in items]
        metadatas = [dict(record.metadata) for record in items]
        try:
            self._collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
        except Exception as exc:  # noqa: BLE001 - no raw chromadb error may escape
            raise VectorIndexError(f"chroma upsert failed: {exc}") from exc

    def query(
        self,
        vector: EmbeddingVector,
        *,
        top_k: int,
        where: VectorFilter | None = None,
    ) -> list[VectorMatch]:
        self._require_dim(vector)
        if top_k <= 0:
            return []
        try:
            result = self._collection.query(
                query_embeddings=[list(vector.values)],
                n_results=top_k,
                where=_to_chroma_where(where),
            )
            ids = (result.get("ids") or [[]])[0]
            distances = (result.get("distances") or [[]])[0]
            matches = [
                VectorMatch(chunk_id=cid, score=1.0 - float(dist))
                for cid, dist in zip(ids, distances)
            ]
        except Exception as exc:  # noqa: BLE001
            raise VectorIndexError(f"chroma query failed: {exc}") from exc
        # Deterministic ordering, matching InMemoryVectorIndex.
        matches.sort(key=lambda match: (-match.score, match.chunk_id))
        return matches

    def delete(self, chunk_ids: Sequence[str]) -> None:
        ids = list(chunk_ids)
        if not ids:
            return
        try:
            self._collection.delete(ids=ids)
        except Exception as exc:  # noqa: BLE001
            raise VectorIndexError(f"chroma delete failed: {exc}") from exc

    def snapshot(self) -> tuple[VectorRecord, ...]:
        """Reconstruct the stored records (test/diagnostic parity with the fake)."""

        try:
            result = self._collection.get(include=["embeddings", "metadatas"])
            ids = result.get("ids") or []
            embeddings = result.get("embeddings") or []
            metadatas = result.get("metadatas") or []
            records = [
                VectorRecord(
                    chunk_id=cid,
                    vector=EmbeddingVector(
                        values=tuple(float(x) for x in embeddings[i]),
                        dim=self._dim,
                        model_id=self._model_id,
                    ),
                    metadata=dict(metadatas[i] or {}),
                )
                for i, cid in enumerate(ids)
            ]
        except Exception as exc:  # noqa: BLE001
            raise VectorIndexError(f"chroma get failed: {exc}") from exc
        return tuple(records)

    def _require_dim(self, vector: EmbeddingVector) -> None:
        if vector.dim != self._dim:
            raise EmbeddingDimensionMismatchError(
                f"index expects dim={self._dim} but got a vector of dim={vector.dim}"
            )

    @classmethod
    def from_config(cls, config: VectorConfig, *, client: Any | None = None) -> "ChromaVectorIndex":
        """Build from a chroma-backed :class:`VectorConfig` (lazily importing chromadb).

        ``client`` may be injected (tests); otherwise a real client is constructed from
        ``config.mode``. Names the collection via ``config.collection_name()`` (6C-3A)
        and pins cosine space.
        """

        if config.backend is not VectorBackend.CHROMA:
            raise ChromaConfigurationError("ChromaVectorIndex requires backend=chroma")
        if client is None:
            client = _build_chroma_client(config)
        try:
            collection = client.get_or_create_collection(
                name=config.collection_name(),
                metadata={"hnsw:space": config.distance},
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorIndexError(f"failed to get/create chroma collection: {exc}") from exc
        return cls(collection, dim=config.dim, model_id=config.embedding_model_id)


def _to_chroma_where(where: VectorFilter | None) -> dict | None:
    """Translate a :class:`VectorFilter` into a Chroma ``where`` clause (AND of $eq)."""

    if where is None or not where.equals:
        return None
    clauses = [{key: {"$eq": value}} for key, value in where.equals.items()]
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _build_chroma_client(config: VectorConfig) -> Any:
    try:
        import chromadb  # noqa: PLC0415 - lazy optional dependency
    except ImportError as exc:
        raise ChromaConfigurationError(
            "chromadb is not installed; install the 'vector' optional dependency"
        ) from exc
    try:
        if config.mode is ChromaMode.EPHEMERAL:
            return chromadb.EphemeralClient()
        if config.mode is ChromaMode.PERSISTENT:
            return chromadb.PersistentClient(path=config.persist_directory)
        return chromadb.HttpClient(host=config.host, port=config.port)
    except Exception as exc:  # noqa: BLE001
        raise VectorIndexError(f"failed to create chroma client: {exc}") from exc
