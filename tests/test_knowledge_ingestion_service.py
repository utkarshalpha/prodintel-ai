"""Phase 6C-2 -- KnowledgeIngestionService: dedup, orphan-safety, resume, recovery.

Mandatory scenarios A-J, all offline (HashEmbeddingClient + InMemoryVectorIndex):

A duplicate ingest -> skipped               F upsert ok / write-back fails -> recover
B embedding failure -> rows NULL            G model mismatch
C upsert failure -> rows NULL               H dimension mismatch
D resume after embedding failure            I no DB txn during embed()
E resume after upsert failure               J batch-boundary recovery

Fault injection uses thin wrappers around the real fakes so a "first run" can fail
and a "resume run" (sharing the same session + index) can succeed.
"""

from __future__ import annotations

import pytest

from app.ai_contracts.knowledge import KnowledgeSourceContract
from app.models.knowledge import KnowledgeChunk, KnowledgeSource
from app.repositories.knowledge_repository import (
    KnowledgeChunkRepository,
    KnowledgeSourceRepository,
)
from app.services.errors import (
    ChunkOrdinalError,
    CorpusEmbeddingModelMismatchError,
    KnowledgeSourceNotFoundError,
    KnowledgeIngestionFailedError,
)
from app.services.knowledge_ingestion_service import KnowledgeIngestionService
from app.vector.errors import EmbeddingClientError, EmbeddingDimensionMismatchError, VectorIndexError
from app.vector.embedding_fake import HashEmbeddingClient
from app.vector.index_fake import InMemoryVectorIndex

from tests.app_helpers import make_engine, make_session_factory

_DIM = 8
_MODEL = "hash-fake-v1"


@pytest.fixture
def session():
    db = make_session_factory(make_engine())()
    try:
        yield db
    finally:
        db.close()


# --------------------------------------------------------------------- builders
def _contract(*, n_chunks=3, corpus_version="pm-corpus-v1", model_id=_MODEL, prefix="rice") -> KnowledgeSourceContract:
    return KnowledgeSourceContract(
        source_type="framework",
        framework="RICE",
        title="RICE Reference",
        corpus_version=corpus_version,
        embedding_model_id=model_id,
        chunks=[
            {"ordinal": i, "content": f"{prefix} chunk content number {i}", "token_count": 5}
            for i in range(n_chunks)
        ],
    )


def _service(session, *, embedder=None, index=None, chunk_repo=None, batch_size=128) -> KnowledgeIngestionService:
    # NB: use explicit `is not None` -- InMemoryVectorIndex defines __len__, so an
    # empty index is falsy and `index or ...` would silently swap in a throwaway one.
    return KnowledgeIngestionService(
        session,
        KnowledgeSourceRepository(session),
        chunk_repo if chunk_repo is not None else KnowledgeChunkRepository(session),
        embedder if embedder is not None else HashEmbeddingClient(dim=_DIM, model_id=_MODEL),
        index if index is not None else InMemoryVectorIndex(dim=_DIM),
        batch_size=batch_size,
    )


def _chroma_ids(session, source_id):
    rows = KnowledgeChunkRepository(session).list_for_source(source_id)
    return [row.chroma_id for row in rows]


# ---------------------------------------------------------------- fault wrappers
class _FaultyEmbedder:
    """Delegates to a real embedder but raises EmbeddingClientError on the Nth embed."""

    def __init__(self, inner, *, fail_on_call):
        self._inner = inner
        self._fail_on_call = fail_on_call
        self._calls = 0

    @property
    def model_id(self):
        return self._inner.model_id

    @property
    def dim(self):
        return self._inner.dim

    def embed_one(self, text):
        return self._inner.embed_one(text)

    def embed(self, texts):
        self._calls += 1
        if self._calls == self._fail_on_call:
            raise EmbeddingClientError("simulated embedding outage")
        return self._inner.embed(texts)


class _FaultyIndex:
    """Delegates to a real index but raises VectorIndexError on every upsert."""

    def __init__(self, inner):
        self._inner = inner

    @property
    def dim(self):
        return self._inner.dim

    def upsert(self, records):
        raise VectorIndexError("simulated vector-store outage")

    def query(self, *a, **k):
        return self._inner.query(*a, **k)

    def delete(self, ids):
        return self._inner.delete(ids)


class _FaultyWriteBackChunkRepo:
    """Wraps a real chunk repo; raises on set_chroma_ids BEFORE flushing (first N calls)."""

    def __init__(self, inner, *, fail_times=1):
        self._inner = inner
        self._remaining = fail_times

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def set_chroma_ids(self, mapping):
        if self._remaining > 0:
            self._remaining -= 1
            raise RuntimeError("simulated write-back failure")
        return self._inner.set_chroma_ids(mapping)


class _FlushThenFailWriteBackChunkRepo:
    """Delegates set_chroma_ids to inner (so the UPDATE is flushed + chroma_id mutated
    in-memory), THEN raises -- simulating a failure at/after commit, the harder
    identity-map case."""

    def __init__(self, inner, *, fail_times=1):
        self._inner = inner
        self._remaining = fail_times

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def set_chroma_ids(self, mapping):
        result = self._inner.set_chroma_ids(mapping)  # flush happens here
        if self._remaining > 0:
            self._remaining -= 1
            raise RuntimeError("simulated commit failure after flush")
        return result


class _TxnSpyEmbedder:
    """Records session.in_transaction() at each embed call (ADR-010 assertion)."""

    def __init__(self, inner, session):
        self._inner = inner
        self._session = session
        self.in_txn_at_embed = []

    @property
    def model_id(self):
        return self._inner.model_id

    @property
    def dim(self):
        return self._inner.dim

    def embed_one(self, text):
        return self._inner.embed_one(text)

    def embed(self, texts):
        self.in_txn_at_embed.append(self._session.in_transaction())
        return self._inner.embed(texts)


# =============================================================== A: duplicate
def test_A_duplicate_ingest_returns_skipped(session) -> None:
    first = _service(session).ingest(_contract())
    assert first.created is True and first.embedded_count == 3

    second = _service(session).ingest(_contract())
    assert second.created is False
    assert second.skipped is True
    assert second.embedded_count == 0
    assert second.source_id == first.source_id


# =============================================================== B: embed fail
def test_B_embedding_failure_leaves_rows_committed_and_chroma_null(session) -> None:
    embedder = _FaultyEmbedder(HashEmbeddingClient(dim=_DIM, model_id=_MODEL), fail_on_call=1)
    index = InMemoryVectorIndex(dim=_DIM)
    with pytest.raises(KnowledgeIngestionFailedError):
        _service(session, embedder=embedder, index=index).ingest(_contract())

    # Rows are committed; the source exists; every chroma_id is NULL; no vectors.
    sources = KnowledgeSourceRepository(session).list(corpus_version="pm-corpus-v1")
    assert len(sources) == 1
    assert _chroma_ids(session, sources[0].id) == [None, None, None]
    assert len(index) == 0


# =============================================================== C: upsert fail
def test_C_upsert_failure_leaves_rows_committed_and_chroma_null(session) -> None:
    index = InMemoryVectorIndex(dim=_DIM)
    with pytest.raises(KnowledgeIngestionFailedError):
        _service(session, index=_FaultyIndex(index)).ingest(_contract())

    sources = KnowledgeSourceRepository(session).list(corpus_version="pm-corpus-v1")
    assert len(sources) == 1
    assert _chroma_ids(session, sources[0].id) == [None, None, None]
    assert len(index) == 0


# =============================================================== D: resume embed
def test_D_resume_after_embedding_failure_succeeds(session) -> None:
    real = HashEmbeddingClient(dim=_DIM, model_id=_MODEL)
    index = InMemoryVectorIndex(dim=_DIM)
    with pytest.raises(KnowledgeIngestionFailedError):
        _service(session, embedder=_FaultyEmbedder(real, fail_on_call=1), index=index).ingest(_contract())

    # Self-healing: re-ingest with a working embedder resumes the same source.
    result = _service(session, embedder=real, index=index).ingest(_contract())
    assert result.created is False and result.skipped is False
    assert result.embedded_count == 3
    source_id = result.source_id
    assert all(cid is not None for cid in _chroma_ids(session, source_id))
    assert len(index) == 3


# =============================================================== E: resume upsert
def test_E_resume_after_upsert_failure_succeeds(session) -> None:
    index = InMemoryVectorIndex(dim=_DIM)
    with pytest.raises(KnowledgeIngestionFailedError):
        _service(session, index=_FaultyIndex(index)).ingest(_contract())

    result = _service(session, index=index).ingest(_contract())  # working index, same store
    assert result.embedded_count == 3
    assert all(cid is not None for cid in _chroma_ids(session, result.source_id))
    assert len(index) == 3


# =============================================================== F: recovery
def test_F_upsert_succeeds_writeback_fails_then_reembed_recovers(session) -> None:
    index = InMemoryVectorIndex(dim=_DIM)
    real_chunk_repo = KnowledgeChunkRepository(session)
    faulty = _FaultyWriteBackChunkRepo(real_chunk_repo, fail_times=1)

    # Upsert succeeds (vectors land in the index), but write-back fails -> rollback.
    with pytest.raises(RuntimeError):
        _service(session, index=index, chunk_repo=faulty).ingest(_contract())

    sources = KnowledgeSourceRepository(session).list(corpus_version="pm-corpus-v1")
    source_id = sources[0].id
    assert _chroma_ids(session, source_id) == [None, None, None]  # write-back rolled back
    assert len(index) == 3  # vectors survived the successful upsert

    # reembed_pending re-embeds + re-upserts (idempotent, same keys) + writes back.
    result = _service(session, index=index).reembed_pending(source_id)
    assert result.embedded_count == 3
    assert all(cid is not None for cid in _chroma_ids(session, source_id))
    assert len(index) == 3  # vector count unchanged -- no duplication


# =============================================================== G: model mismatch
def test_G_model_mismatch_raises(session) -> None:
    _service(session).ingest(_contract(corpus_version="shared-v1", model_id=_MODEL))

    other_embedder = HashEmbeddingClient(dim=_DIM, model_id="other-model")
    with pytest.raises(CorpusEmbeddingModelMismatchError):
        _service(session, embedder=other_embedder).ingest(
            _contract(corpus_version="shared-v1", model_id="other-model", prefix="different")
        )


# =============================================================== H: dim mismatch
def test_H_dimension_mismatch_raises(session) -> None:
    embedder = HashEmbeddingClient(dim=16, model_id=_MODEL)  # != index dim 8
    index = InMemoryVectorIndex(dim=_DIM)
    with pytest.raises(EmbeddingDimensionMismatchError):
        _service(session, embedder=embedder, index=index).ingest(_contract())

    # Pre-flight: nothing was persisted.
    assert KnowledgeSourceRepository(session).list(corpus_version="pm-corpus-v1") == []


# =============================================================== I: no txn during embed
def test_I_no_db_transaction_held_during_embed(session) -> None:
    spy = _TxnSpyEmbedder(HashEmbeddingClient(dim=_DIM, model_id=_MODEL), session)
    _service(session, embedder=spy, batch_size=2).ingest(_contract(n_chunks=5))
    # 5 chunks, batch_size 2 -> exactly 3 embed calls, each outside any DB transaction.
    assert len(spy.in_txn_at_embed) == 3
    assert all(in_txn is False for in_txn in spy.in_txn_at_embed)


# =============================================================== J: batch boundary
def test_J_batch_boundary_recovery(session) -> None:
    real = HashEmbeddingClient(dim=_DIM, model_id=_MODEL)
    index = InMemoryVectorIndex(dim=_DIM)
    # batch_size 2 over 5 chunks -> embed calls for batches [0,1],[2,3],[4]; fail the 3rd.
    with pytest.raises(KnowledgeIngestionFailedError):
        _service(session, embedder=_FaultyEmbedder(real, fail_on_call=3), index=index, batch_size=2).ingest(
            _contract(n_chunks=5)
        )

    source_id = KnowledgeSourceRepository(session).list(corpus_version="pm-corpus-v1")[0].id
    chroma = _chroma_ids(session, source_id)
    assert [c is not None for c in chroma] == [True, True, True, True, False]  # batch 3 (last chunk) pending
    assert len(index) == 4

    # Resume completes the final batch.
    result = _service(session, embedder=real, index=index, batch_size=2).ingest(_contract(n_chunks=5))
    assert result.embedded_count == 1
    assert all(c is not None for c in _chroma_ids(session, source_id))
    assert len(index) == 5


# ---------------------------------------------------- reembed_pending edge cases
def test_reembed_pending_missing_source_raises(session) -> None:
    import uuid

    with pytest.raises(KnowledgeSourceNotFoundError):
        _service(session).reembed_pending(uuid.uuid4())


def test_reembed_pending_on_complete_source_is_skip(session) -> None:
    result = _service(session).ingest(_contract())
    reembed = _service(session).reembed_pending(result.source_id)
    assert reembed.skipped is True and reembed.embedded_count == 0


# ----------------------------------------------------- ordinal validation (4/6)
def _ordinal_contract(ordinals) -> KnowledgeSourceContract:
    return KnowledgeSourceContract(
        source_type="framework",
        framework="RICE",
        title="RICE Reference",
        corpus_version="pm-corpus-v1",
        embedding_model_id=_MODEL,
        chunks=[
            {"ordinal": o, "content": f"body number {i}", "token_count": 3}
            for i, o in enumerate(ordinals)
        ],
    )


def test_duplicate_ordinals_raise(session) -> None:
    with pytest.raises(ChunkOrdinalError):
        _service(session).ingest(_ordinal_contract([0, 0, 1]))
    assert KnowledgeSourceRepository(session).list(corpus_version="pm-corpus-v1") == []


def test_gapped_ordinals_raise(session) -> None:
    with pytest.raises(ChunkOrdinalError):
        _service(session).ingest(_ordinal_contract([0, 2, 3]))


def test_unordered_but_contiguous_ordinals_are_accepted(session) -> None:
    # [2, 0, 1] is contiguous 0-based once sorted -> valid; the service sorts internally.
    result = _service(session).ingest(_ordinal_contract([2, 0, 1]))
    assert result.created is True and result.embedded_count == 3


# ------------------------------------- write-back commit failure after flush (3)
def test_writeback_failure_after_flush_rolls_back_and_resumes(session) -> None:
    index = InMemoryVectorIndex(dim=_DIM)
    faulty = _FlushThenFailWriteBackChunkRepo(KnowledgeChunkRepository(session), fail_times=1)

    # Upsert succeeds; set_chroma_ids flushes the UPDATE; then commit fails.
    with pytest.raises(RuntimeError):
        _service(session, index=index, chunk_repo=faulty).ingest(_contract())

    source_id = KnowledgeSourceRepository(session).list(corpus_version="pm-corpus-v1")[0].id
    # rollback + expire_all must discard the flushed chroma_id so resume sees them pending.
    assert _chroma_ids(session, source_id) == [None, None, None]

    result = _service(session, index=index).reembed_pending(source_id)
    assert result.embedded_count == 3
    assert all(cid is not None for cid in _chroma_ids(session, source_id))
    assert len(index) == 3  # idempotent re-upsert -> no duplication
