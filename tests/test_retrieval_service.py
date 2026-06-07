"""Phase 6D-iii -- RetrievalService end-to-end, offline & deterministic.

Drives the REAL KnowledgeIngestionService to populate a corpus (rows + vectors via
HashEmbeddingClient + InMemoryVectorIndex), then retrieves. Covers ranking, top-k,
metadata filtering, citation correctness, min_score, empty results, stale vectors,
error wrapping, the model-mismatch guard, ADR-010, determinism, and the count
partition.
"""

from __future__ import annotations

import uuid

import pytest

from app.ai_contracts.knowledge import KnowledgeSourceContract
from app.ai_contracts.retrieval import RetrievalQuery
from app.repositories.knowledge_repository import (
    KnowledgeChunkRepository,
    KnowledgeSourceRepository,
)
from app.services.errors import CorpusEmbeddingModelMismatchError, RetrievalFailedError
from app.services.knowledge_ingestion_service import KnowledgeIngestionService
from app.services.retrieval_service import RetrievalService
from app.vector.errors import EmbeddingClientError, VectorIndexError
from app.vector.embedding_fake import HashEmbeddingClient
from app.vector.index_fake import InMemoryVectorIndex
from app.vector.interfaces import VectorRecord

from tests.app_helpers import make_engine, make_session_factory

_DIM = 8
_MODEL = "hash-fake-v1"
_CORPUS = "pm-corpus-v1"


@pytest.fixture
def session():
    db = make_session_factory(make_engine())()
    try:
        yield db
    finally:
        db.close()


def _ingest(session, embedder, index, *, chunks, corpus_version=_CORPUS):
    """chunks: list of (content, framework|None). Returns the created source id."""

    contract = KnowledgeSourceContract(
        source_type="framework",
        framework="RICE",
        title="RICE Reference",
        author="Sean Ellis",
        source_license="CC-BY",
        uri="https://example.com/rice",
        corpus_version=corpus_version,
        embedding_model_id=embedder.model_id,
        chunks=[
            {"ordinal": i, "content": content, "token_count": 5, "framework": framework}
            for i, (content, framework) in enumerate(chunks)
        ],
    )
    svc = KnowledgeIngestionService(
        session,
        KnowledgeSourceRepository(session),
        KnowledgeChunkRepository(session),
        embedder,
        index,
    )
    return svc.ingest(contract).source_id


def _service(session, embedder, index) -> RetrievalService:
    return RetrievalService(
        session,
        embedder,
        index,
        KnowledgeChunkRepository(session),
        KnowledgeSourceRepository(session),
    )


def _seed(session, *, chunks):
    embedder = HashEmbeddingClient(dim=_DIM, model_id=_MODEL)
    index = InMemoryVectorIndex(dim=_DIM)
    source_id = _ingest(session, embedder, index, chunks=chunks)
    return embedder, index, source_id


_DEFAULT_CHUNKS = [("alpha rice", "RICE"), ("beta rice", "RICE"), ("gamma jtbd", "JTBD")]


# --- 1. Ranking correctness ------------------------------------------------
def test_ranking_exact_match_first(session) -> None:
    embedder, index, _ = _seed(session, chunks=_DEFAULT_CHUNKS)
    result = _service(session, embedder, index).retrieve(
        RetrievalQuery(text="alpha rice", corpus_version=_CORPUS)
    )
    assert result.citations[0].content == "alpha rice"
    assert result.citations[0].score == pytest.approx(1.0, abs=1e-9)
    scores = [c.score for c in result.citations]
    assert scores == sorted(scores, reverse=True)


# --- 2. Top-k behavior -----------------------------------------------------
def test_top_k_bounds_result_count(session) -> None:
    embedder, index, _ = _seed(session, chunks=_DEFAULT_CHUNKS)
    result = _service(session, embedder, index).retrieve(
        RetrievalQuery(text="alpha rice", corpus_version=_CORPUS, top_k=2)
    )
    assert result.meta.vector_match_count == 2
    assert len(result.citations) == 2


# --- 3. Metadata filtering -------------------------------------------------
def test_framework_filter(session) -> None:
    embedder, index, _ = _seed(session, chunks=_DEFAULT_CHUNKS)
    svc = _service(session, embedder, index)

    rice = svc.retrieve(RetrievalQuery(text="alpha rice", corpus_version=_CORPUS, framework="RICE"))
    assert {c.framework.value for c in rice.citations} == {"RICE"}
    assert len(rice.citations) == 2

    jtbd = _service(session, embedder, index).retrieve(
        RetrievalQuery(text="alpha rice", corpus_version=_CORPUS, framework="JTBD")
    )
    assert [c.framework.value for c in jtbd.citations] == ["JTBD"]


# --- 4. Citation correctness -----------------------------------------------
def test_citation_fields_from_system_of_record(session) -> None:
    embedder, index, source_id = _seed(session, chunks=_DEFAULT_CHUNKS)
    result = _service(session, embedder, index).retrieve(
        RetrievalQuery(text="alpha rice", corpus_version=_CORPUS, framework="RICE")
    )
    top = result.citations[0]
    assert top.content == "alpha rice"  # authoritative text from Postgres
    assert top.source_id == source_id
    assert top.ordinal == 0
    assert top.framework.value == "RICE"
    assert top.corpus_version == _CORPUS
    assert top.source_title == "RICE Reference"
    assert top.source_type.value == "framework"
    assert top.source_uri == "https://example.com/rice"
    assert top.source_license == "CC-BY"


# --- 5. min_score filtering ------------------------------------------------
def test_min_score_drops_weak_matches(session) -> None:
    embedder, index, _ = _seed(session, chunks=_DEFAULT_CHUNKS)
    result = _service(session, embedder, index).retrieve(
        RetrievalQuery(text="alpha rice", corpus_version=_CORPUS, min_score=0.999)
    )
    # Only the exact-match chunk clears a 0.999 cosine floor.
    assert [c.content for c in result.citations] == ["alpha rice"]
    assert result.meta.dropped_score_count == result.meta.resolved_count - 1
    assert result.meta.resolved_count >= 1


# --- 6. Empty results ------------------------------------------------------
def test_unknown_corpus_returns_empty(session) -> None:
    embedder, index, _ = _seed(session, chunks=_DEFAULT_CHUNKS)
    result = _service(session, embedder, index).retrieve(
        RetrievalQuery(text="alpha rice", corpus_version="does-not-exist")
    )
    assert result.citations == [] and result.meta.vector_match_count == 0


def test_filter_matching_nothing_returns_empty(session) -> None:
    embedder, index, _ = _seed(session, chunks=_DEFAULT_CHUNKS)
    result = _service(session, embedder, index).retrieve(
        RetrievalQuery(text="alpha rice", corpus_version=_CORPUS, framework="MoSCoW")
    )
    assert result.citations == [] and result.meta.vector_match_count == 0


# --- 7. Stale vector handling ----------------------------------------------
def test_stale_vector_is_dropped(session) -> None:
    embedder, index, _ = _seed(session, chunks=_DEFAULT_CHUNKS)
    # Inject an orphan vector (valid uuid, no row) + a malformed-id vector.
    query_vec = embedder.embed_one("alpha rice")
    index.upsert([
        VectorRecord(chunk_id=str(uuid.uuid4()), vector=query_vec, metadata={"corpus_version": _CORPUS}),
        VectorRecord(chunk_id="not-a-uuid", vector=query_vec, metadata={"corpus_version": _CORPUS}),
    ])
    result = _service(session, embedder, index).retrieve(
        RetrievalQuery(text="alpha rice", corpus_version=_CORPUS, top_k=10)
    )
    assert result.meta.dropped_stale_count == 2
    assert result.meta.resolved_count == 3  # the three real chunks
    assert all(c.content in {"alpha rice", "beta rice", "gamma jtbd"} for c in result.citations)


# --- 8. Error wrapping -----------------------------------------------------
class _FailingEmbedder:
    model_id = _MODEL
    dim = _DIM

    def embed_one(self, text):
        raise EmbeddingClientError("embed down")

    def embed(self, texts):  # pragma: no cover
        raise EmbeddingClientError("embed down")


class _FailingIndex:
    dim = _DIM

    def query(self, *a, **k):
        raise VectorIndexError("index down")

    def upsert(self, *a, **k):  # pragma: no cover
        pass

    def delete(self, *a, **k):  # pragma: no cover
        pass


def test_embedding_failure_wrapped(session) -> None:
    embedder, index, _ = _seed(session, chunks=_DEFAULT_CHUNKS)
    svc = RetrievalService(
        session, _FailingEmbedder(), index,
        KnowledgeChunkRepository(session), KnowledgeSourceRepository(session),
    )
    with pytest.raises(RetrievalFailedError) as ei:
        svc.retrieve(RetrievalQuery(text="alpha rice", corpus_version=_CORPUS))
    assert isinstance(ei.value.__cause__, EmbeddingClientError)


def test_vector_query_failure_wrapped(session) -> None:
    embedder, index, _ = _seed(session, chunks=_DEFAULT_CHUNKS)
    svc = RetrievalService(
        session, embedder, _FailingIndex(),
        KnowledgeChunkRepository(session), KnowledgeSourceRepository(session),
    )
    with pytest.raises(RetrievalFailedError) as ei:
        svc.retrieve(RetrievalQuery(text="alpha rice", corpus_version=_CORPUS))
    assert isinstance(ei.value.__cause__, VectorIndexError)


# --- 9. Model mismatch guard -----------------------------------------------
def test_model_mismatch_raises(session) -> None:
    _, index, _ = _seed(session, chunks=_DEFAULT_CHUNKS)  # corpus embedded by hash-fake-v1
    other = HashEmbeddingClient(dim=_DIM, model_id="other-model")
    svc = _service(session, other, index)
    with pytest.raises(CorpusEmbeddingModelMismatchError):
        svc.retrieve(RetrievalQuery(text="alpha rice", corpus_version=_CORPUS))


# --- 10. ADR-010 verification ----------------------------------------------
class _TxnSpy:
    """Wraps embedder + index, recording session.in_transaction() at each call."""

    def __init__(self, embedder, index, session):
        self._embedder = embedder
        self._index = index
        self._session = session
        self.in_txn = []

    # embedder surface
    @property
    def model_id(self):
        return self._embedder.model_id

    @property
    def dim(self):
        return self._embedder.dim

    def embed_one(self, text):
        self.in_txn.append(self._session.in_transaction())
        return self._embedder.embed_one(text)

    def embed(self, texts):  # pragma: no cover
        return self._embedder.embed(texts)

    # index surface
    def query(self, *a, **k):
        self.in_txn.append(self._session.in_transaction())
        return self._index.query(*a, **k)

    def upsert(self, *a, **k):  # pragma: no cover
        return self._index.upsert(*a, **k)

    def delete(self, *a, **k):  # pragma: no cover
        return self._index.delete(*a, **k)


def test_no_db_transaction_during_embed_and_query(session) -> None:
    embedder, index, _ = _seed(session, chunks=_DEFAULT_CHUNKS)
    spy = _TxnSpy(embedder, index, session)
    RetrievalService(
        session, spy, spy, KnowledgeChunkRepository(session), KnowledgeSourceRepository(session)
    ).retrieve(RetrievalQuery(text="alpha rice", corpus_version=_CORPUS))
    assert spy.in_txn == [False, False]  # embed call, then query call -- neither in a txn


# --- 11. Deterministic retrieval -------------------------------------------
def test_retrieval_is_deterministic(session) -> None:
    embedder, index, _ = _seed(session, chunks=_DEFAULT_CHUNKS)
    svc = _service(session, embedder, index)
    first = svc.retrieve(RetrievalQuery(text="alpha rice", corpus_version=_CORPUS))
    second = _service(session, embedder, index).retrieve(
        RetrievalQuery(text="alpha rice", corpus_version=_CORPUS)
    )
    assert first == second


class _DuplicatingIndex:
    """A misbehaving backend that returns every match twice (adversarial-pass guard)."""

    def __init__(self, inner):
        self._inner = inner
        self.dim = inner.dim

    def query(self, *a, **k):
        out = list(self._inner.query(*a, **k))
        return out + out  # duplicate every match

    def upsert(self, *a, **k):  # pragma: no cover
        return self._inner.upsert(*a, **k)

    def delete(self, *a, **k):  # pragma: no cover
        return self._inner.delete(*a, **k)


def test_duplicate_index_matches_are_normalized(session) -> None:
    embedder, index, _ = _seed(session, chunks=_DEFAULT_CHUNKS)
    svc = RetrievalService(
        session, embedder, _DuplicatingIndex(index),
        KnowledgeChunkRepository(session), KnowledgeSourceRepository(session),
    )
    result = svc.retrieve(RetrievalQuery(text="alpha rice", corpus_version=_CORPUS, top_k=10))

    chunk_ids = [c.chunk_id for c in result.citations]
    assert len(chunk_ids) == len(set(chunk_ids))  # no duplicate citations
    assert result.meta.vector_match_count == len(result.citations) == 3  # normalized to distinct
    m = result.meta
    assert m.resolved_count == m.vector_match_count - m.dropped_stale_count
    assert len(result.citations) == m.resolved_count - m.dropped_score_count


# --- 12. Count-partition correctness ---------------------------------------
def test_count_partition_holds(session) -> None:
    embedder, index, _ = _seed(session, chunks=_DEFAULT_CHUNKS)
    index.upsert([
        VectorRecord(chunk_id=str(uuid.uuid4()), vector=embedder.embed_one("alpha rice"),
                     metadata={"corpus_version": _CORPUS}),
    ])
    result = _service(session, embedder, index).retrieve(
        RetrievalQuery(text="alpha rice", corpus_version=_CORPUS, top_k=10, min_score=0.999)
    )
    m = result.meta
    assert m.resolved_count == m.vector_match_count - m.dropped_stale_count
    assert len(result.citations) == m.resolved_count - m.dropped_score_count
    assert m.dropped_stale_count == 1  # the injected orphan
