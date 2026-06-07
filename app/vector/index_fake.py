"""Deterministic, in-memory vector index for tests (Phase 6C).

:class:`InMemoryVectorIndex` stores records in an insertion-ordered dict keyed by
``chunk_id``, so ``upsert`` is idempotent (re-assigning a key overwrites in place)
and iteration order is stable. Similarity is dot-product -- which equals cosine
because inputs are L2-normalized -- and results are deterministically ordered by
``(score desc, chunk_id asc)``. ``snapshot()`` exposes the stored records (in
insertion order) so tests can assert exactly what was upserted.

No network, no ChromaDB: this is the vector-store analog of the in-memory SQLite
used elsewhere in the test suite.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.vector.errors import EmbeddingDimensionMismatchError
from app.vector.interfaces import EmbeddingVector, VectorFilter, VectorMatch, VectorRecord

__all__ = ["InMemoryVectorIndex"]


class InMemoryVectorIndex:
    """A deterministic in-memory :class:`~app.vector.interfaces.VectorIndex`."""

    def __init__(self, *, dim: int) -> None:
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self._dim = dim
        self._store: dict[str, VectorRecord] = {}

    @property
    def dim(self) -> int:
        return self._dim

    def upsert(self, records: Sequence[VectorRecord]) -> None:
        for record in records:
            self._require_dim(record.vector)
            # Re-assigning an existing key overwrites in place and preserves order.
            self._store[record.chunk_id] = record

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

        equals = where.equals if where is not None else {}
        matches: list[VectorMatch] = []
        for record in self._store.values():
            if not self._matches_filter(record, equals):
                continue
            score = sum(a * b for a, b in zip(vector.values, record.vector.values))
            matches.append(VectorMatch(chunk_id=record.chunk_id, score=score))

        # Deterministic: highest score first, ties broken by ascending chunk_id.
        matches.sort(key=lambda match: (-match.score, match.chunk_id))
        return matches[:top_k]

    def delete(self, chunk_ids: Sequence[str]) -> None:
        for chunk_id in chunk_ids:
            self._store.pop(chunk_id, None)  # missing ids ignored

    def snapshot(self) -> tuple[VectorRecord, ...]:
        """The stored records in insertion order (for test assertions)."""

        return tuple(self._store.values())

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, chunk_id: object) -> bool:
        return chunk_id in self._store

    # ------------------------------------------------------------------ helpers
    def _require_dim(self, vector: EmbeddingVector) -> None:
        if vector.dim != self._dim:
            raise EmbeddingDimensionMismatchError(
                f"index expects dim={self._dim} but got a vector of dim={vector.dim}"
            )

    @staticmethod
    def _matches_filter(record: VectorRecord, equals: dict) -> bool:
        return all(record.metadata.get(key) == value for key, value in equals.items())
