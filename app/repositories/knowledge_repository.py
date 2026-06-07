"""Repositories for the RAG knowledge corpus (Phase 6B).

Thin persistence over a SQLAlchemy :class:`Session`, following the same conventions
as every other repository: the session is injected, writes ``flush`` but never
``commit`` (the service owns the unit-of-work), eager reads use ``selectinload``,
list queries are deterministically ordered, and there is no business logic --
hashing, the source roll-up, embedding, dedup *policy*, and transaction control all
live in the Phase 6C ingestion service.

Two design points specific to this corpus (both approved in the Phase 6B review):

* ``KnowledgeSourceRepository.list`` is **lazy** -- it does not eager-load chunks.
  A single source can own thousands of chunks, so loading them for a listing would
  be a memory/latency cliff. :meth:`KnowledgeSourceRepository.get_with_chunks` is the
  explicit, single-source eager path.
* ``chroma_id`` write-back uses the **ORM unit-of-work** (per-object assignment +
  flush), never a Core bulk ``update()``. This keeps the field-selective
  immutability guard on :class:`KnowledgeChunk` authoritative: the only permitted
  mutation in this whole layer is ``chroma_id``, and it flows through the ORM so the
  ``before_update`` guard runs.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.ai_contracts.enums import FrameworkName, KnowledgeSourceType
from app.models.knowledge import KnowledgeChunk, KnowledgeSource

__all__ = ["KnowledgeSourceRepository", "KnowledgeChunkRepository"]


class KnowledgeSourceRepository:
    """Read/insert access to :class:`KnowledgeSource` (immutable; never updated)."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, source: KnowledgeSource) -> KnowledgeSource:
        """Insert a source (and any attached chunks via cascade) and flush."""

        self._session.add(source)
        self._session.flush()
        return source

    def create_with_chunks(
        self, source: KnowledgeSource, chunks: Iterable[KnowledgeChunk]
    ) -> KnowledgeSource:
        """Insert a source together with its chunks in one unit, then flush.

        The chunks persist via the ``all, delete-orphan`` cascade on
        ``KnowledgeSource.chunks``. This is the primary ingestion insert path.
        """

        for chunk in chunks:
            source.chunks.append(chunk)
        self._session.add(source)
        self._session.flush()
        return source

    def get(self, source_id: uuid.UUID) -> KnowledgeSource | None:
        """Fetch a source by primary key. Does **not** load chunks (lazy)."""

        return self._session.get(KnowledgeSource, source_id)

    def get_with_chunks(self, source_id: uuid.UUID) -> KnowledgeSource | None:
        """Fetch a source with its chunks eagerly loaded (ordinal-ordered).

        The explicit single-source eager path. Returns ``None`` if absent.
        """

        stmt = (
            select(KnowledgeSource)
            .where(KnowledgeSource.id == source_id)
            .options(selectinload(KnowledgeSource.chunks))
        )
        return self._session.scalars(stmt).unique().one_or_none()

    def get_by_corpus_version_and_content_hash(
        self, corpus_version: str, content_hash: str
    ) -> KnowledgeSource | None:
        """Dedupe lookup: the source for this (corpus_version, content_hash), or None.

        Backed by ``uq_knowledge_source_corpus_version_content_hash`` so at most one
        row can match. ``content_hash`` is treated as opaque -- the service guarantees
        it encodes the embedding model (the model-drift guard).
        """

        stmt = select(KnowledgeSource).where(
            KnowledgeSource.corpus_version == corpus_version,
            KnowledgeSource.content_hash == content_hash,
        )
        return self._session.scalar(stmt)

    def list(
        self,
        *,
        corpus_version: str | None = None,
        source_type: KnowledgeSourceType | None = None,
        framework: FrameworkName | None = None,
        limit: int = 100,
    ) -> Sequence[KnowledgeSource]:
        """List sources, optionally filtered. **Lazy**: chunks are not loaded.

        Ordered most-recent-first. Use :meth:`get_with_chunks` to load one source's
        chunks explicitly.
        """

        stmt = select(KnowledgeSource)
        if corpus_version is not None:
            stmt = stmt.where(KnowledgeSource.corpus_version == corpus_version)
        if source_type is not None:
            stmt = stmt.where(KnowledgeSource.source_type == source_type)
        if framework is not None:
            stmt = stmt.where(KnowledgeSource.framework == framework)
        stmt = stmt.order_by(KnowledgeSource.created_at.desc()).limit(limit)
        return list(self._session.scalars(stmt))


class KnowledgeChunkRepository:
    """Read/insert access to :class:`KnowledgeChunk`.

    The only mutation this repository performs is the ``chroma_id`` write-back, via
    the ORM so the chunk immutability guard stays authoritative.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def add_all(self, chunks: Iterable[KnowledgeChunk]) -> Sequence[KnowledgeChunk]:
        """Insert chunks and flush. Secondary insert path (resume / standalone)."""

        items = list(chunks)
        if not items:
            return []
        self._session.add_all(items)
        self._session.flush()
        return items

    def get(self, chunk_id: uuid.UUID) -> KnowledgeChunk | None:
        return self._session.get(KnowledgeChunk, chunk_id)

    def get_by_source_and_hash(
        self, source_id: uuid.UUID, content_hash: str
    ) -> KnowledgeChunk | None:
        """Per-chunk idempotency/resume lookup, scoped to one source.

        Backed by ``uq_knowledge_chunk_source_id_content_hash``.
        """

        stmt = select(KnowledgeChunk).where(
            KnowledgeChunk.source_id == source_id,
            KnowledgeChunk.content_hash == content_hash,
        )
        return self._session.scalar(stmt)

    def list_for_source(self, source_id: uuid.UUID) -> Sequence[KnowledgeChunk]:
        """All chunks for a source, ordinal-ordered."""

        stmt = (
            select(KnowledgeChunk)
            .where(KnowledgeChunk.source_id == source_id)
            .order_by(KnowledgeChunk.ordinal)
        )
        return list(self._session.scalars(stmt))

    def list_unembedded(self, source_id: uuid.UUID) -> Sequence[KnowledgeChunk]:
        """Committed chunks for a source still missing a vector (``chroma_id IS NULL``).

        Resume support for the orphan-safe ingestion flow: re-embed exactly these.
        Ordinal-ordered.
        """

        stmt = (
            select(KnowledgeChunk)
            .where(KnowledgeChunk.source_id == source_id, KnowledgeChunk.chroma_id.is_(None))
            .order_by(KnowledgeChunk.ordinal)
        )
        return list(self._session.scalars(stmt))

    def set_chroma_id(self, chunk_id: uuid.UUID, chroma_id: uuid.UUID) -> KnowledgeChunk:
        """Write back a single chunk's vector handle (the one permitted mutation).

        Per-object ORM assignment + flush, so the immutability guard runs and confirms
        only ``chroma_id`` changed. Raises :class:`ValueError` if the chunk is absent.
        """

        chunk = self._session.get(KnowledgeChunk, chunk_id)
        if chunk is None:
            raise ValueError(f"KnowledgeChunk {chunk_id} not found")
        chunk.chroma_id = chroma_id
        self._session.flush()
        return chunk

    def set_chroma_ids(self, mapping: Mapping[uuid.UUID, uuid.UUID]) -> int:
        """Batch ``chroma_id`` write-back. Returns the number of chunks updated.

        Lenient on missing ids (skips them) so the caller can detect drift via the
        returned count. One flush for the whole batch; each assignment goes through
        the ORM so the immutability guard remains authoritative.
        """

        if not mapping:
            return 0
        updated = 0
        for chunk_id, chroma_id in mapping.items():
            chunk = self._session.get(KnowledgeChunk, chunk_id)
            if chunk is None:
                continue
            chunk.chroma_id = chroma_id
            updated += 1
        self._session.flush()
        return updated
