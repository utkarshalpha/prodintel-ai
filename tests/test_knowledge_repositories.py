"""Phase 6B -- KnowledgeSourceRepository / KnowledgeChunkRepository.

Covers the approved review's eleven verification requirements against an in-memory
SQLite database (FK pragma ON, mirroring production Postgres semantics):

1. flush-not-commit            7. set_chroma_id
2. dedupe lookups              8. set_chroma_ids
3. get_with_chunks eager       9. immutability guard still enforced
4. list() stays lazy          10. empty-input safety
5. ordinal ordering           11. cascade persistence via create_with_chunks
6. list_unembedded

Reads use ``session.expunge_all()`` after commit so each query genuinely hits the
database (otherwise identity-map hits would mask eager/lazy differences). Eager-vs-
lazy is asserted two ways: the SQLAlchemy ``inspect(obj).unloaded`` set, and a
statement counter proving an eager access issues no further SQL.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager

import pytest
from sqlalchemy import event, inspect as sa_inspect

from app.ai_contracts.enums import FrameworkName, KnowledgeSourceType
from app.models.knowledge import KnowledgeChunk, KnowledgeChunkImmutableError, KnowledgeSource
from app.repositories.knowledge_repository import (
    KnowledgeChunkRepository,
    KnowledgeSourceRepository,
)

from tests.app_helpers import make_engine, make_session_factory


@pytest.fixture
def session():
    db = make_session_factory(make_engine())()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def count_statements(session):
    """Count SQL statements executed on the session's engine within the block."""

    bind = session.get_bind()
    counter = {"n": 0}

    def _on_execute(_conn, _cursor, _statement, _params, _context, _executemany):
        counter["n"] += 1

    event.listen(bind, "after_cursor_execute", _on_execute)
    try:
        yield counter
    finally:
        event.remove(bind, "after_cursor_execute", _on_execute)


def _source(*, content_hash: str, corpus_version: str = "pm-corpus-v1",
            source_type: str = "framework", framework: str | None = "RICE",
            title: str = "RICE Prioritization Reference") -> KnowledgeSource:
    return KnowledgeSource(
        source_type=source_type,
        framework=framework,
        title=title,
        corpus_version=corpus_version,
        embedding_model_id="hash-fake-v1",
        content_hash=content_hash,
    )


def _chunk(source: KnowledgeSource, ordinal: int, *, embedded: bool = False) -> KnowledgeChunk:
    return KnowledgeChunk(
        corpus_version=source.corpus_version,
        embedding_model_id=source.embedding_model_id,
        ordinal=ordinal,
        content=f"chunk body number {ordinal}",
        content_hash=f"{ordinal:064d}",
        token_count=4,
        chroma_id=uuid.uuid4() if embedded else None,
    )


# 1 -------------------------------------------------------------------------
def test_create_with_chunks_flushes_not_commits(session) -> None:
    repo = KnowledgeSourceRepository(session)
    source = _source(content_hash="f" * 64)
    repo.create_with_chunks(source, [_chunk(source, 0), _chunk(source, 1)])

    assert source.id is not None  # flush populated the PK
    source_id = source.id

    session.rollback()  # nothing was committed -> rollback discards it
    assert repo.get(source_id) is None


# 2 -------------------------------------------------------------------------
def test_dedupe_lookups(session) -> None:
    src_repo = KnowledgeSourceRepository(session)
    chunk_repo = KnowledgeChunkRepository(session)
    source = _source(content_hash="a" * 64)
    src_repo.create_with_chunks(source, [_chunk(source, 0), _chunk(source, 1)])
    source_id = source.id
    session.commit()
    session.expunge_all()

    hit = src_repo.get_by_corpus_version_and_content_hash("pm-corpus-v1", "a" * 64)
    assert hit is not None and hit.id == source_id
    # Same hash, different corpus snapshot -> miss (the version scopes the dedupe).
    assert src_repo.get_by_corpus_version_and_content_hash("pm-corpus-v2", "a" * 64) is None
    # Different hash -> miss.
    assert src_repo.get_by_corpus_version_and_content_hash("pm-corpus-v1", "b" * 64) is None

    assert chunk_repo.get_by_source_and_hash(source_id, f"{0:064d}") is not None
    assert chunk_repo.get_by_source_and_hash(source_id, "z" * 64) is None


# 3 + 5 ---------------------------------------------------------------------
def test_get_with_chunks_eager_and_ordinal_ordered(session) -> None:
    repo = KnowledgeSourceRepository(session)
    source = _source(content_hash="c" * 64)
    # Insert out of ordinal order; the eager load must return them sorted.
    repo.create_with_chunks(source, [_chunk(source, 2), _chunk(source, 0), _chunk(source, 1)])
    source_id = source.id
    session.commit()
    session.expunge_all()

    loaded = repo.get_with_chunks(source_id)
    assert loaded is not None
    assert "chunks" not in sa_inspect(loaded).unloaded  # eagerly loaded

    with count_statements(session) as counter:
        ordinals = [chunk.ordinal for chunk in loaded.chunks]
    assert counter["n"] == 0, "accessing eagerly-loaded chunks must issue no SQL"
    assert ordinals == [0, 1, 2]


# 4 -------------------------------------------------------------------------
def test_list_is_lazy_and_filters(session) -> None:
    repo = KnowledgeSourceRepository(session)
    framework_src = _source(content_hash="1" * 64, source_type="framework", framework="RICE")
    repo.create_with_chunks(framework_src, [_chunk(framework_src, 0)])
    book_src = _source(content_hash="2" * 64, source_type="book", framework=None, title="Inspired")
    repo.create_with_chunks(book_src, [_chunk(book_src, 0)])
    session.commit()
    session.expunge_all()

    everything = repo.list()
    assert len(everything) == 2
    for src in everything:
        assert "chunks" in sa_inspect(src).unloaded, "list() must not eager-load chunks"

    books = repo.list(source_type=KnowledgeSourceType.BOOK)
    assert [s.source_type for s in books] == [KnowledgeSourceType.BOOK]

    rice = repo.list(framework=FrameworkName.RICE)
    assert len(rice) == 1 and rice[0].framework is FrameworkName.RICE


# 6 -------------------------------------------------------------------------
def test_list_unembedded_returns_only_null_chroma_ordered(session) -> None:
    repo = KnowledgeSourceRepository(session)
    chunk_repo = KnowledgeChunkRepository(session)
    source = _source(content_hash="6" * 64)
    repo.create_with_chunks(
        source,
        [
            _chunk(source, 0, embedded=False),
            _chunk(source, 1, embedded=True),
            _chunk(source, 2, embedded=False),
            _chunk(source, 3, embedded=True),
        ],
    )
    source_id = source.id
    session.commit()
    session.expunge_all()

    pending = chunk_repo.list_unembedded(source_id)
    assert [c.ordinal for c in pending] == [0, 2]
    assert all(c.chroma_id is None for c in pending)


# 7 + 9 ---------------------------------------------------------------------
def test_set_chroma_id_writes_back_and_guard_stays_enforced(session) -> None:
    repo = KnowledgeSourceRepository(session)
    chunk_repo = KnowledgeChunkRepository(session)
    source = _source(content_hash="7" * 64)
    repo.create_with_chunks(source, [_chunk(source, 0)])
    chunk_id = source.chunks[0].id
    session.commit()
    session.expunge_all()

    handle = uuid.uuid4()
    updated = chunk_repo.set_chroma_id(chunk_id, handle)  # ORM update -> guard allows chroma_id
    assert updated.chroma_id == handle
    session.commit()
    session.expunge_all()
    assert chunk_repo.get(chunk_id).chroma_id == handle

    # The guard is still authoritative: any non-chroma_id mutation is rejected.
    chunk = chunk_repo.get(chunk_id)
    chunk.content = "tampered body that no longer matches its vector"
    with pytest.raises(KnowledgeChunkImmutableError):
        session.flush()


def test_set_chroma_id_raises_on_missing_chunk(session) -> None:
    with pytest.raises(ValueError):
        KnowledgeChunkRepository(session).set_chroma_id(uuid.uuid4(), uuid.uuid4())


# 8 -------------------------------------------------------------------------
def test_set_chroma_ids_batch_counts_and_is_lenient(session) -> None:
    repo = KnowledgeSourceRepository(session)
    chunk_repo = KnowledgeChunkRepository(session)
    source = _source(content_hash="8" * 64)
    repo.create_with_chunks(source, [_chunk(source, 0), _chunk(source, 1)])
    ids = [c.id for c in source.chunks]
    session.commit()
    session.expunge_all()

    mapping = {
        ids[0]: uuid.uuid4(),
        ids[1]: uuid.uuid4(),
        uuid.uuid4(): uuid.uuid4(),  # missing id -> skipped, not counted
    }
    count = chunk_repo.set_chroma_ids(mapping)
    assert count == 2

    session.commit()
    session.expunge_all()
    for chunk_id in ids:
        assert chunk_repo.get(chunk_id).chroma_id == mapping[chunk_id]


# 10 ------------------------------------------------------------------------
def test_empty_input_safety(session) -> None:
    chunk_repo = KnowledgeChunkRepository(session)
    assert chunk_repo.add_all([]) == []
    assert chunk_repo.set_chroma_ids({}) == 0


# 11 ------------------------------------------------------------------------
def test_cascade_persistence_through_create_with_chunks(session) -> None:
    repo = KnowledgeSourceRepository(session)
    chunk_repo = KnowledgeChunkRepository(session)
    source = _source(content_hash="d" * 64)
    repo.create_with_chunks(source, [_chunk(source, 0), _chunk(source, 1), _chunk(source, 2)])
    source_id = source.id
    session.commit()
    session.expunge_all()

    loaded = repo.get_with_chunks(source_id)
    assert len(loaded.chunks) == 3
    assert all(c.source_id == source_id for c in loaded.chunks)
    # Reachable via the chunk repo too, ordinal-ordered.
    by_source = chunk_repo.list_for_source(source_id)
    assert [c.ordinal for c in by_source] == [0, 1, 2]
