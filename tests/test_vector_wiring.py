"""Phase 6C-3C -- vector infrastructure wiring, offline via doubles.

Covers memory + chroma factory dispatch, attach_vector_infra (app.state), startup
validation (dim consistency, read probe), provider independence, and the lazy-import
guarantee. No chromadb, no network.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from app.vector import (
    EmbeddingClient,
    HashEmbeddingClient,
    InMemoryVectorIndex,
    VectorIndex,
)
from app.vector.config import VectorBackend, VectorConfig
from app.vector.embedding_chroma import ChromaEmbeddingAdapter
from app.vector.errors import ChromaConfigurationError
from app.vector.index_chroma import ChromaVectorIndex
from app.vector.wiring import (
    DEFAULT_MEMORY_DIM,
    attach_vector_infra,
    make_embedding_client,
    make_vector_index,
    validate_vector_infra,
)


# ----------------------------------------------------------------- doubles
class _Collection:
    def __init__(self):
        self.metadata = {"hnsw:space": "cosine"}

    def upsert(self, ids, embeddings, metadatas):  # pragma: no cover - unused here
        pass

    def query(self, query_embeddings, n_results, where=None):
        return {"ids": [[]], "distances": [[]]}

    def delete(self, ids):  # pragma: no cover
        pass

    def get(self, include=None):  # pragma: no cover
        return {"ids": [], "embeddings": [], "metadatas": []}


class _Client:
    def __init__(self):
        self.created: list = []

    def get_or_create_collection(self, name, metadata=None):
        self.created.append((name, metadata))
        return _Collection()


class _ExplodingCollection:
    def query(self, **_k):
        raise RuntimeError("vector store down")


class _DoubleEF:
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, inputs):
        return [[1.0] + [0.0] * (self.dim - 1) for _ in inputs]


def _chroma_config(dim=4, model_id="all-MiniLM-L6-v2"):
    return VectorConfig.from_env(
        {
            "VECTOR_BACKEND": "chroma",
            "CHROMA_MODE": "ephemeral",
            "EMBEDDING_MODEL_ID": model_id,
            "EMBEDDING_DIM": str(dim),
        }
    )


def _app():
    return SimpleNamespace(state=SimpleNamespace())


# --- 1. Memory backend wiring ----------------------------------------------
def test_memory_factories_default_dim() -> None:
    cfg = VectorConfig.from_env({})  # memory, no dim pinned
    embedder = make_embedding_client(cfg)
    index = make_vector_index(cfg)
    assert isinstance(embedder, HashEmbeddingClient)
    assert isinstance(index, InMemoryVectorIndex)
    assert embedder.dim == DEFAULT_MEMORY_DIM == index.dim
    assert embedder.model_id == "hash-fake-v1"


def test_memory_factories_honor_config_dim() -> None:
    cfg = VectorConfig(backend=VectorBackend.MEMORY, dim=32, embedding_model_id="custom")
    embedder = make_embedding_client(cfg)
    index = make_vector_index(cfg)
    assert embedder.dim == 32 == index.dim
    assert embedder.model_id == "custom"


# --- 2. Chroma backend wiring (doubles) ------------------------------------
def test_chroma_factories_use_adapters_and_naming() -> None:
    cfg = _chroma_config(dim=4)
    embedder = make_embedding_client(cfg, embedding_function=_DoubleEF(4))
    client = _Client()
    index = make_vector_index(cfg, client=client)

    assert isinstance(embedder, ChromaEmbeddingAdapter)
    assert isinstance(index, ChromaVectorIndex)
    assert embedder.dim == 4 and embedder.model_id == "all-MiniLM-L6-v2"
    name, metadata = client.created[0]
    assert name == cfg.collection_name()
    assert metadata == {"hnsw:space": "cosine"}


# --- 3. attach_vector_infra ------------------------------------------------
def test_attach_memory_sets_app_state() -> None:
    app = _app()
    embedder, index = attach_vector_infra(app, VectorConfig.from_env({}))
    assert app.state.embedding_client is embedder
    assert app.state.vector_index is index
    assert isinstance(app.state.embedding_client, EmbeddingClient)
    assert isinstance(app.state.vector_index, VectorIndex)


def test_attach_chroma_with_doubles_sets_app_state() -> None:
    app = _app()
    attach_vector_infra(app, _chroma_config(4), client=_Client(), embedding_function=_DoubleEF(4))
    assert isinstance(app.state.embedding_client, ChromaEmbeddingAdapter)
    assert isinstance(app.state.vector_index, ChromaVectorIndex)


# --- 4. Startup validation -------------------------------------------------
def test_validate_dim_mismatch_raises() -> None:
    cfg = VectorConfig.from_env({})  # memory
    with pytest.raises(ChromaConfigurationError):
        validate_vector_infra(HashEmbeddingClient(dim=8), InMemoryVectorIndex(dim=4), cfg)


def test_validate_config_dim_mismatch_raises() -> None:
    cfg = VectorConfig(backend=VectorBackend.MEMORY, dim=8)
    with pytest.raises(ChromaConfigurationError):
        validate_vector_infra(HashEmbeddingClient(dim=4), InMemoryVectorIndex(dim=4), cfg)


def test_validate_chroma_probe_failure_raises() -> None:
    cfg = _chroma_config(4)
    index = ChromaVectorIndex(_ExplodingCollection(), dim=4, model_id="m")
    with pytest.raises(ChromaConfigurationError):
        validate_vector_infra(HashEmbeddingClient(dim=4, model_id="m"), index, cfg)


def test_validate_chroma_probe_success() -> None:
    cfg = _chroma_config(4)
    index = ChromaVectorIndex(_Collection(), dim=4, model_id="m")
    validate_vector_infra(HashEmbeddingClient(dim=4, model_id="m"), index, cfg)  # must not raise


def test_validate_memory_runs_no_probe() -> None:
    cfg = VectorConfig.from_env({})
    # Memory path: dim consistent, no probe -> no error even with a plain in-memory index.
    validate_vector_infra(HashEmbeddingClient(dim=16), InMemoryVectorIndex(dim=16), cfg)


# --- 5. Lazy-import guarantee ----------------------------------------------
def test_wiring_does_not_import_chromadb() -> None:
    import app.vector.wiring  # noqa: F401

    assert "chromadb" not in sys.modules


# --- provider independence -------------------------------------------------
def test_attached_objects_satisfy_protocols() -> None:
    app = _app()
    attach_vector_infra(app, _chroma_config(4), client=_Client(), embedding_function=_DoubleEF(4))
    assert isinstance(app.state.embedding_client, EmbeddingClient)
    assert isinstance(app.state.vector_index, VectorIndex)
