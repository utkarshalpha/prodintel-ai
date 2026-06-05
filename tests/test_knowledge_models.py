"""Phase 6A -- ORM behavior for the knowledge corpus: immutability + ownership.

Exercises the two model invariants directly against an in-memory SQLite database
(FK pragma on, mirroring production Postgres semantics):

* ``KnowledgeSource`` is fully immutable (any update raises).
* ``KnowledgeChunk`` is immutable **except** ``chroma_id`` (the post-embedding
  write-back path), and deleting a source cascades to its chunks.
"""

from __future__ import annotations

import uuid

import pytest

from app.models.knowledge import (
    KnowledgeChunk,
    KnowledgeChunkImmutableError,
    KnowledgeSource,
    KnowledgeSourceImmutableError,
)

from tests.app_helpers import make_engine, make_session_factory


@pytest.fixture
def session():
    db = make_session_factory(make_engine())()
    try:
        yield db
    finally:
        db.close()


def _make_source(**overrides) -> KnowledgeSource:
    fields = dict(
        source_type="framework",
        framework="RICE",
        title="RICE Prioritization Reference",
        corpus_version="pm-corpus-v1",
        embedding_model_id="hash-fake-v1",
        content_hash="0" * 64,
    )
    fields.update(overrides)
    return KnowledgeSource(**fields)


def _make_chunk(source: KnowledgeSource, **overrides) -> KnowledgeChunk:
    fields = dict(
        corpus_version=source.corpus_version,
        embedding_model_id=source.embedding_model_id,
        ordinal=0,
        content="Reach times Impact times Confidence divided by Effort.",
        content_hash="1" * 64,
        token_count=9,
    )
    fields.update(overrides)
    chunk = KnowledgeChunk(**fields)
    chunk.source = source
    return chunk


def _persist(session, *objs):
    session.add_all(objs)
    session.commit()


def test_knowledge_source_is_immutable(session) -> None:
    source = _make_source()
    _persist(session, source)

    source.title = "Tampered title"
    with pytest.raises(KnowledgeSourceImmutableError):
        session.commit()


def test_knowledge_chunk_content_is_immutable(session) -> None:
    source = _make_source()
    chunk = _make_chunk(source)
    _persist(session, source, chunk)

    chunk.content = "rewritten body that no longer matches its vector"
    with pytest.raises(KnowledgeChunkImmutableError):
        session.commit()


def test_knowledge_chunk_chroma_id_is_writable(session) -> None:
    """The one permitted mutation: writing back the vector handle after embedding."""

    source = _make_source()
    chunk = _make_chunk(source, chroma_id=None)
    _persist(session, source, chunk)
    assert chunk.chroma_id is None

    handle = uuid.uuid4()
    chunk.chroma_id = handle
    session.commit()  # must NOT raise

    session.expire_all()
    reloaded = session.get(KnowledgeChunk, chunk.id)
    assert reloaded.chroma_id == handle


def test_deleting_source_cascades_to_chunks(session) -> None:
    source = _make_source()
    chunk_a = _make_chunk(source, ordinal=0, content_hash="a" * 64)
    chunk_b = _make_chunk(source, ordinal=1, content="Effort is measured in person-months.", content_hash="b" * 64)
    _persist(session, source, chunk_a, chunk_b)
    chunk_ids = [chunk_a.id, chunk_b.id]

    session.delete(source)
    session.commit()

    assert session.get(KnowledgeSource, source.id) is None
    for chunk_id in chunk_ids:
        assert session.get(KnowledgeChunk, chunk_id) is None


def test_source_owns_chunks_in_ordinal_order(session) -> None:
    source = _make_source()
    # Insert out of order; the relationship is ordered by ordinal.
    _make_chunk(source, ordinal=2, content="third", content_hash="c" * 64)
    _make_chunk(source, ordinal=0, content="first", content_hash="d" * 64)
    _make_chunk(source, ordinal=1, content="second", content_hash="e" * 64)
    _persist(session, source)

    session.expire_all()
    reloaded = session.get(KnowledgeSource, source.id)
    assert [chunk.ordinal for chunk in reloaded.chunks] == [0, 1, 2]
