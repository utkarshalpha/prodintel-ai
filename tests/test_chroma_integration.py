"""Phase 6C-3C -- gated real-ChromaDB integration tests.

These run against a REAL chromadb in-process (``EphemeralClient``: no network, no
Docker, no disk), and confirm that the pure-Python doubles used in the offline suite
faithfully mirror real Chroma. They are double-gated so they never run in the default
suite:

* skipped unless ``chromadb`` is importable, and
* skipped unless ``RUN_CHROMA_TESTS=1`` is set.

The embedding model is the deterministic ``HashEmbeddingClient`` (no model download);
only the *vector index* is exercised against real Chroma. A real
SentenceTransformer adapter would require a model download and is intentionally out of
scope here.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

_HAS_CHROMADB = importlib.util.find_spec("chromadb") is not None
_RUN = os.environ.get("RUN_CHROMA_TESTS") == "1"

pytestmark = pytest.mark.skipif(
    not (_HAS_CHROMADB and _RUN),
    reason="real-chroma integration: requires chromadb installed and RUN_CHROMA_TESTS=1",
)

from app.vector import HashEmbeddingClient, VectorFilter, VectorRecord  # noqa: E402
from app.vector.config import VectorConfig  # noqa: E402
from app.vector.wiring import attach_vector_infra, make_vector_index, validate_vector_infra  # noqa: E402

_DIM = 8
_EMB = HashEmbeddingClient(dim=_DIM, model_id="integration-test")


def _config(prefix="prodintel_it"):
    return VectorConfig.from_env(
        {
            "VECTOR_BACKEND": "chroma",
            "CHROMA_MODE": "ephemeral",
            "CHROMA_COLLECTION_PREFIX": prefix,
            "EMBEDDING_MODEL_ID": "integration-test",
            "EMBEDDING_DIM": str(_DIM),
        }
    )


def _record(chunk_id, text, **metadata):
    return VectorRecord(chunk_id=chunk_id, vector=_EMB.embed_one(text), metadata=metadata)


def test_real_chroma_round_trip() -> None:
    index = make_vector_index(_config())  # real EphemeralClient
    index.upsert([
        _record("a", "alpha", corpus_version="v1", framework="RICE"),
        _record("b", "beta", corpus_version="v1", framework="JTBD"),
    ])

    matches = index.query(_EMB.embed_one("alpha"), top_k=2)
    assert matches[0].chunk_id == "a"  # exact match ranks first
    assert matches[0].score >= matches[-1].score

    filtered = index.query(_EMB.embed_one("alpha"), top_k=5, where=VectorFilter(equals={"framework": "JTBD"}))
    assert [m.chunk_id for m in filtered] == ["b"]

    snap = {r.chunk_id: r.metadata for r in index.snapshot()}
    assert snap["a"]["corpus_version"] == "v1" and snap["a"]["framework"] == "RICE"

    index.delete(["a"])
    assert "a" not in {r.chunk_id for r in index.snapshot()}


def test_real_chroma_upsert_is_idempotent() -> None:
    index = make_vector_index(_config(prefix="prodintel_idem"))
    rec = _record("c1", "alpha", corpus_version="v1")
    index.upsert([rec])
    index.upsert([rec])  # same chunk_id -> overwrite, not duplicate
    assert len(index.snapshot()) == 1


def test_real_chroma_collection_named_and_cosine() -> None:
    config = _config(prefix="prodintel_named")
    index = make_vector_index(config)
    # The collection the adapter is bound to carries the derived name + cosine space.
    assert index._collection.name == config.collection_name()
    assert index._collection.metadata.get("hnsw:space") == "cosine"


def test_real_chroma_startup_validation_and_attach() -> None:
    from types import SimpleNamespace

    config = _config(prefix="prodintel_attach")
    # Inject a deterministic embedding function so attach builds no real model
    # (no download); the index is a REAL ephemeral chroma collection (client=None).
    def _ef(inputs):
        return [list(_EMB.embed_one(text).values) for text in inputs]

    app = SimpleNamespace(state=SimpleNamespace())
    embedder, index = attach_vector_infra(app, config, embedding_function=_ef)
    assert app.state.embedding_client is embedder and app.state.vector_index is index

    # The attached pair round-trips through real chroma (attach already ran the probe).
    vector = embedder.embed_one("alpha")
    index.upsert([VectorRecord(chunk_id="z", vector=vector, metadata={"corpus_version": "v1"})])
    assert index.query(vector, top_k=1)[0].chunk_id == "z"
