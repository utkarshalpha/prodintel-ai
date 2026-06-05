"""SQLAlchemy models for the RAG knowledge corpus (Phase 6 -- RAG Foundations).

* ``KnowledgeSource`` is an ingested external knowledge artifact -- a framework
  reference, a PM book, a historical decision record, a company strategy document --
  identified by its category (``source_type``), an optional specific ``framework``,
  and a content hash. It is the corpus analog of the immutable :class:`Signal` root:
  an ORM event blocks any update.
* ``KnowledgeChunk`` is one retrievable span of a source. It owns the canonical text
  (the relational store is the system of record; ChromaDB only holds vectors,
  reachable via the nullable ``chroma_id`` handle). ``ON DELETE CASCADE`` from the
  source side, so deleting a source removes its chunks.

Immutability is enforced at two granularities:

* ``KnowledgeSource`` is fully immutable (mirrors :class:`Signal`).
* ``KnowledgeChunk`` is immutable **except** for ``chroma_id``: the vector handle is
  written back after embedding, so a field-selective ``before_update`` guard permits
  exactly that one column to change and rejects any other mutation -- a bare UNIQUE
  constraint cannot stop ``UPDATE content=...``, which would silently invalidate the
  chunk's vector.

Versioning columns (``corpus_version``, ``embedding_model_id``) are pinned on both
tables so any future retrieval is reproducible (which corpus snapshot, which
embedder). The enum *type names* are distinct (``knowledge_source_type``,
``knowledge_source_framework_name``, ``knowledge_chunk_framework_name``) so they do
not collide with each other or with enums created in earlier revisions -- the same
disambiguation used for the decision and conflict enums.

This module adds tables only: repositories, the ingestion service, and the vector
infrastructure (ChromaDB, embedding/index clients) are later Phase 6 sub-phases.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Enum as SAEnum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    event,
    inspect as sa_inspect,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.ai_contracts.enums import FrameworkName, KnowledgeSourceType
from app.db.base import Base

__all__ = [
    "KnowledgeSource",
    "KnowledgeChunk",
    "KnowledgeSourceImmutableError",
    "KnowledgeChunkImmutableError",
]


# The single column a chunk is permitted to mutate after creation: the vector handle
# written back once embedding completes. Everything else is immutable.
_CHUNK_MUTABLE_FIELDS = frozenset({"chroma_id"})


class KnowledgeSourceImmutableError(Exception):
    """Raised when code attempts to update an immutable :class:`KnowledgeSource`."""


class KnowledgeChunkImmutableError(Exception):
    """Raised when code attempts to mutate a :class:`KnowledgeChunk` field other than
    ``chroma_id``."""


def _utcnow() -> datetime:
    """Timezone-aware current timestamp (client-side default)."""

    return datetime.now(timezone.utc)


def _value_enum(enum_cls, name: str) -> SAEnum:
    """Portable enum column storing the enum *values* (e.g. 'RICE', 'MoSCoW').

    Using ``values_callable`` makes the stored strings match the contract/API values
    rather than the enum member names -- and, critically for :class:`FrameworkName`,
    preserves the canonical mixed-case spelling.
    """

    return SAEnum(enum_cls, name=name, values_callable=lambda e: [member.value for member in e])


class KnowledgeSource(Base):
    """Immutable, ingested external knowledge artifact -- a corpus source."""

    __tablename__ = "knowledge_source"
    __table_args__ = (
        # Idempotent re-ingest within a corpus snapshot: the same source content under
        # the same corpus_version is one row, not many.
        UniqueConstraint(
            "corpus_version", "content_hash", name="uq_knowledge_source_corpus_version_content_hash"
        ),
        Index("ix_knowledge_source_corpus_version", "corpus_version"),
        Index("ix_knowledge_source_source_type", "source_type"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    # Reserved for the Workspace phase: nullable, no FK yet.
    workspace_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, nullable=True)
    source_type: Mapped[KnowledgeSourceType] = mapped_column(
        _value_enum(KnowledgeSourceType, "knowledge_source_type"), nullable=False
    )
    framework: Mapped[FrameworkName | None] = mapped_column(
        _value_enum(FrameworkName, "knowledge_source_framework_name"), nullable=True
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    author: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_license: Mapped[str | None] = mapped_column(String(128), nullable=True)
    uri: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    corpus_version: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding_model_id: Mapped[str] = mapped_column(String(128), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow, nullable=False)

    chunks: Mapped[list["KnowledgeChunk"]] = relationship(
        back_populates="source",
        cascade="all, delete-orphan",
        order_by="KnowledgeChunk.ordinal",
    )


class KnowledgeChunk(Base):
    """One retrievable span of a :class:`KnowledgeSource`.

    The ``id`` is the stable handle used as the vector id in ChromaDB; ``chroma_id``
    is written back once the vector is upserted (and is the only mutable column).
    """

    __tablename__ = "knowledge_chunk"
    __table_args__ = (
        # Per-source dedup: identical text within a source is one chunk. Scoped to the
        # source so the same boilerplate appearing in two books stays two traceable rows.
        UniqueConstraint("source_id", "content_hash", name="uq_knowledge_chunk_source_id_content_hash"),
        # FK is not the PK, so index it for source -> chunks loads and the CASCADE.
        Index("ix_knowledge_chunk_source_id", "source_id"),
        # Chroma hit -> row lookup.
        Index("ix_knowledge_chunk_chroma_id", "chroma_id"),
        Index("ix_knowledge_chunk_corpus_version", "corpus_version"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    source_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("knowledge_source.id", ondelete="CASCADE"),
        nullable=False,
    )
    # Denormalized from the parent at build time so a chunk row is self-describing
    # for retrieval/filtering without a join.
    corpus_version: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding_model_id: Mapped[str] = mapped_column(String(128), nullable=False)
    framework: Mapped[FrameworkName | None] = mapped_column(
        _value_enum(FrameworkName, "knowledge_chunk_framework_name"), nullable=True
    )
    section: Mapped[str | None] = mapped_column(String(255), nullable=True)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    # The ChromaDB vector handle: NULL until the chunk is embedded and upserted.
    chroma_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow, nullable=False)

    source: Mapped[KnowledgeSource] = relationship(back_populates="chunks")


@event.listens_for(KnowledgeSource, "before_update", propagate=True)
def _block_knowledge_source_update(_mapper, _connection, _target: KnowledgeSource) -> None:
    """Enforce knowledge-source immutability at the ORM layer (mirrors :class:`Signal`)."""

    raise KnowledgeSourceImmutableError("KnowledgeSource rows are immutable and cannot be updated")


@event.listens_for(KnowledgeChunk, "before_update", propagate=True)
def _block_knowledge_chunk_mutation(_mapper, _connection, target: KnowledgeChunk) -> None:
    """Permit only ``chroma_id`` to change; reject any other field mutation.

    The chunk must stay writable for the post-embedding ``chroma_id`` write-back, so
    a plain immutability block is too strict and a UNIQUE constraint is too weak
    (it would not stop a ``content`` rewrite). This guard inspects per-column change
    history and raises if anything other than ``chroma_id`` is dirty.
    """

    state = sa_inspect(target)
    changed = {
        attr.key for attr in state.mapper.column_attrs if state.attrs[attr.key].history.has_changes()
    }
    forbidden = changed - _CHUNK_MUTABLE_FIELDS
    if forbidden:
        raise KnowledgeChunkImmutableError(
            "KnowledgeChunk fields are immutable except chroma_id; "
            f"attempted to change: {sorted(forbidden)}"
        )
