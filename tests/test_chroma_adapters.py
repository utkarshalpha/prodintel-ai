"""Phase 6C-3B -- ChromaVectorIndex / ChromaEmbeddingAdapter, offline via doubles.

No chromadb, no network, no Docker. Pure-Python doubles stand in for a Chroma
collection / client / embedding function so the adapters' translation logic, error
wrapping, and naming integration are verified deterministically.
"""

from __future__ import annotations

import math
import sys

import pytest

from app.vector import (
    EmbeddingClient,
    EmbeddingDimensionMismatchError,
    EmbeddingVector,
    EmbeddingClientError,
    HashEmbeddingClient,
    VectorFilter,
    VectorIndex,
    VectorIndexError,
    VectorRecord,
)
from app.vector.config import VectorConfig
from app.vector.embedding_chroma import ChromaEmbeddingAdapter
from app.vector.index_chroma import ChromaVectorIndex


# ---------------------------------------------------------------- doubles
class _RecordingCollection:
    """In-memory stand-in for a chromadb Collection."""

    def __init__(self):
        self.store: dict[str, tuple[list[float], dict]] = {}
        self.upsert_calls: list = []
        self.delete_calls: list = []
        self.query_calls: list = []

    def upsert(self, ids, embeddings, metadatas):
        self.upsert_calls.append((ids, embeddings, metadatas))
        for i, cid in enumerate(ids):
            self.store[cid] = (list(embeddings[i]), dict(metadatas[i]))

    def query(self, query_embeddings, n_results, where=None):
        self.query_calls.append((query_embeddings, n_results, where))
        q = query_embeddings[0]
        scored = []
        for cid, (emb, meta) in self.store.items():
            if not _meta_matches(where, meta):
                continue
            scored.append((_cosine_distance(q, emb), cid))
        scored.sort(key=lambda t: (t[0], t[1]))
        scored = scored[:n_results]
        return {"ids": [[cid for _, cid in scored]], "distances": [[d for d, _ in scored]]}

    def delete(self, ids):
        self.delete_calls.append(ids)
        for cid in ids:
            self.store.pop(cid, None)

    def get(self, include=None):
        ids = list(self.store.keys())
        return {
            "ids": ids,
            "embeddings": [self.store[c][0] for c in ids],
            "metadatas": [self.store[c][1] for c in ids],
        }


class _ExplodingCollection:
    def upsert(self, **_k):
        raise RuntimeError("boom-upsert")

    def query(self, **_k):
        raise RuntimeError("boom-query")

    def delete(self, **_k):
        raise RuntimeError("boom-delete")

    def get(self, **_k):
        raise RuntimeError("boom-get")


class _FakeClient:
    def __init__(self):
        self.created: list = []

    def get_or_create_collection(self, name, metadata=None):
        self.created.append((name, metadata))
        return _RecordingCollection()


class _DoubleEF:
    """Deterministic embedding-function double returning fixed-dim vectors."""

    def __init__(self, dim):
        self.dim = dim
        self.calls: list = []

    def __call__(self, inputs):
        self.calls.append(list(inputs))
        out = []
        for text in inputs:
            seed = sum(ord(ch) for ch in text) or 1
            out.append([float((seed * (i + 1)) % 7 + 1) for i in range(self.dim)])
        return out


class _ExplodingEF:
    def __call__(self, _inputs):
        raise RuntimeError("provider-down")


def _cosine_distance(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - dot / (na * nb)


def _meta_matches(where, meta):
    if not where:
        return True
    if "$and" in where:
        return all(_meta_matches(clause, meta) for clause in where["$and"])
    for key, cond in where.items():
        value = cond["$eq"] if isinstance(cond, dict) and "$eq" in cond else cond
        if meta.get(key) != value:
            return False
    return True


_EMB = HashEmbeddingClient(dim=4, model_id="test-model")


def _record(chunk_id, text, **metadata):
    return VectorRecord(chunk_id=chunk_id, vector=_EMB.embed_one(text), metadata=metadata)


# ----------------------------------------------------- protocol conformance
def test_adapters_satisfy_protocols() -> None:
    assert isinstance(ChromaVectorIndex(_RecordingCollection(), dim=4), VectorIndex)
    assert isinstance(ChromaEmbeddingAdapter(_DoubleEF(4), model_id="m", dim=4), EmbeddingClient)


# ----------------------------------------------------------- upsert translation
def test_upsert_translation() -> None:
    collection = _RecordingCollection()
    index = ChromaVectorIndex(collection, dim=4, model_id="test-model")
    rec = _record("c1", "alpha", corpus_version="v1", ordinal=0, framework="RICE")
    index.upsert([rec])

    ids, embeddings, metadatas = collection.upsert_calls[0]
    assert ids == ["c1"]
    assert embeddings == [list(rec.vector.values)]  # tuple -> list, text never sent
    assert metadatas == [{"corpus_version": "v1", "ordinal": 0, "framework": "RICE"}]


def test_upsert_empty_is_noop() -> None:
    collection = _RecordingCollection()
    ChromaVectorIndex(collection, dim=4).upsert([])
    assert collection.upsert_calls == []


# ------------------------------------------------------------ query translation
def test_query_translation_orders_and_scores() -> None:
    collection = _RecordingCollection()
    index = ChromaVectorIndex(collection, dim=4, model_id="test-model")
    exact = _EMB.embed_one("alpha")
    index.upsert([
        VectorRecord(chunk_id="a", vector=exact, metadata={"corpus_version": "v1"}),
        VectorRecord(chunk_id="b", vector=_EMB.embed_one("faraway"), metadata={"corpus_version": "v1"}),
    ])

    matches = index.query(exact, top_k=2)
    assert [m.chunk_id for m in matches] == ["a", "b"]  # exact match first
    assert matches[0].score == pytest.approx(1.0, abs=1e-9)  # cosine 1 -> score 1
    assert matches[0].score >= matches[1].score


def test_query_top_k_zero_returns_empty() -> None:
    index = ChromaVectorIndex(_RecordingCollection(), dim=4)
    assert index.query(_EMB.embed_one("x"), top_k=0) == []


# ----------------------------------------------------------- metadata filtering
def test_query_metadata_filter_round_trip() -> None:
    collection = _RecordingCollection()
    index = ChromaVectorIndex(collection, dim=4, model_id="test-model")
    index.upsert([
        _record("r", "rice", corpus_version="v1", framework="RICE"),
        _record("j", "jtbd", corpus_version="v1", framework="JTBD"),
    ])

    out = index.query(
        _EMB.embed_one("q"), top_k=10, where=VectorFilter(equals={"framework": "RICE"})
    )
    assert [m.chunk_id for m in out] == ["r"]
    # The translated where clause reached the collection.
    assert collection.query_calls[-1][2] == {"framework": {"$eq": "RICE"}}


# ----------------------------------------------------------- delete translation
def test_delete_translation() -> None:
    collection = _RecordingCollection()
    index = ChromaVectorIndex(collection, dim=4, model_id="test-model")
    index.upsert([_record("x", "x", corpus_version="v1")])
    index.delete(["x", "missing"])
    assert collection.delete_calls == [["x", "missing"]]
    assert "x" not in collection.store


def test_delete_empty_is_noop() -> None:
    collection = _RecordingCollection()
    ChromaVectorIndex(collection, dim=4).delete([])
    assert collection.delete_calls == []


# --------------------------------------------------- snapshot + metadata round-trip
def test_snapshot_round_trip() -> None:
    collection = _RecordingCollection()
    index = ChromaVectorIndex(collection, dim=4, model_id="test-model")
    rec = _record("c1", "alpha", corpus_version="v1", ordinal=3, framework="RICE")
    index.upsert([rec])

    snap = index.snapshot()
    assert len(snap) == 1
    assert snap[0].chunk_id == "c1"
    assert snap[0].metadata == {"corpus_version": "v1", "ordinal": 3, "framework": "RICE"}
    assert snap[0].vector.values == rec.vector.values  # round-trips exactly
    assert snap[0].vector.model_id == "test-model"


# --------------------------------------------------------- typed error wrapping
def test_index_wraps_provider_errors() -> None:
    index = ChromaVectorIndex(_ExplodingCollection(), dim=4, model_id="m")
    rec = _record("x", "x", corpus_version="v1")
    with pytest.raises(VectorIndexError) as ei:
        index.upsert([rec])
    assert isinstance(ei.value.__cause__, RuntimeError)  # no raw chromadb error escapes
    with pytest.raises(VectorIndexError):
        index.query(_EMB.embed_one("x"), top_k=1)
    with pytest.raises(VectorIndexError):
        index.delete(["x"])
    with pytest.raises(VectorIndexError):
        index.snapshot()


def test_embedding_wraps_provider_errors() -> None:
    adapter = ChromaEmbeddingAdapter(_ExplodingEF(), model_id="m", dim=4)
    with pytest.raises(EmbeddingClientError) as ei:
        adapter.embed(["x"])
    assert isinstance(ei.value.__cause__, RuntimeError)


# --------------------------------------------------------- dimension mismatch
def test_index_rejects_wrong_dim_vector() -> None:
    index = ChromaVectorIndex(_RecordingCollection(), dim=4, model_id="m")
    wrong = HashEmbeddingClient(dim=8).embed_one("x")
    with pytest.raises(EmbeddingDimensionMismatchError):
        index.upsert([VectorRecord(chunk_id="x", vector=wrong, metadata={"corpus_version": "v1"})])
    with pytest.raises(EmbeddingDimensionMismatchError):
        index.query(wrong, top_k=1)


def test_embedding_rejects_wrong_provider_dim() -> None:
    adapter = ChromaEmbeddingAdapter(_DoubleEF(8), model_id="m", dim=4)  # EF emits dim 8
    with pytest.raises(EmbeddingDimensionMismatchError):
        adapter.embed(["x"])


# ---------------------------------------------------- embedding behavior
def test_embedding_normalizes_and_preserves_order() -> None:
    adapter = ChromaEmbeddingAdapter(_DoubleEF(4), model_id="test-model", dim=4)
    out = adapter.embed(["a", "b", "c"])
    assert len(out) == 3
    for vector in out:
        assert vector.model_id == "test-model" and vector.dim == 4
        assert math.isclose(math.sqrt(sum(x * x for x in vector.values)), 1.0, abs_tol=1e-9)
    assert adapter.embed(["a", "b"])[0].values == adapter.embed(["b", "a"])[1].values
    assert adapter.embed_one("a").values == adapter.embed(["a"])[0].values


def test_embedding_empty_batch() -> None:
    assert ChromaEmbeddingAdapter(_DoubleEF(4), model_id="m", dim=4).embed([]) == []


# ----------------------------------------------------- collection naming integration
def test_from_config_uses_collection_name_and_cosine_space() -> None:
    config = VectorConfig.from_env(
        {
            "VECTOR_BACKEND": "chroma",
            "CHROMA_MODE": "ephemeral",
            "EMBEDDING_MODEL_ID": "all-MiniLM-L6-v2",
            "EMBEDDING_DIM": "4",
        }
    )
    client = _FakeClient()
    index = ChromaVectorIndex.from_config(config, client=client)

    name, metadata = client.created[0]
    assert name == config.collection_name()
    assert metadata == {"hnsw:space": "cosine"}
    assert index.dim == 4


def test_from_config_rejects_non_chroma_backend() -> None:
    from app.vector.errors import ChromaConfigurationError

    memory = VectorConfig.from_env({})
    with pytest.raises(ChromaConfigurationError):
        ChromaVectorIndex.from_config(memory, client=_FakeClient())
    with pytest.raises(ChromaConfigurationError):
        ChromaEmbeddingAdapter.from_config(memory, embedding_function=_DoubleEF(4))


# ------------------------------------------------------------ lazy-import guarantee
def test_adapters_do_not_import_chromadb() -> None:
    import app.vector.embedding_chroma  # noqa: F401
    import app.vector.index_chroma  # noqa: F401

    assert "chromadb" not in sys.modules
