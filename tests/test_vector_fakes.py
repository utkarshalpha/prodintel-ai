"""Phase 6C-1 -- deterministic offline vector fakes.

Covers the required test areas: determinism, idempotent upsert, query ordering,
metadata filtering, delete, snapshot, dimension validation, and length/order
parity -- plus Protocol conformance, L2-normalization, DTO immutability, and the
offline guarantee (no ChromaDB import).
"""

from __future__ import annotations

import math
import sys

import pytest
from pydantic import ValidationError

from app.vector import (
    EmbeddingClient,
    EmbeddingDimensionMismatchError,
    EmbeddingVector,
    HashEmbeddingClient,
    InMemoryVectorIndex,
    VectorFilter,
    VectorIndex,
    VectorRecord,
)


# --- Protocol conformance --------------------------------------------------
def test_fakes_satisfy_protocols() -> None:
    assert isinstance(HashEmbeddingClient(dim=4), EmbeddingClient)
    assert isinstance(InMemoryVectorIndex(dim=4), VectorIndex)


# --- Determinism -----------------------------------------------------------
def test_embedding_is_deterministic_across_instances() -> None:
    a = HashEmbeddingClient(dim=16)
    b = HashEmbeddingClient(dim=16)
    assert a.embed_one("RICE prioritization").values == b.embed_one("RICE prioritization").values


def test_different_text_yields_different_vector() -> None:
    client = HashEmbeddingClient(dim=16)
    assert client.embed_one("alpha").values != client.embed_one("beta").values


def test_embedding_is_l2_normalized() -> None:
    vector = HashEmbeddingClient(dim=32).embed_one("normalize me")
    norm = math.sqrt(sum(component * component for component in vector.values))
    assert abs(norm - 1.0) < 1e-9


# --- Length / order parity -------------------------------------------------
def test_embed_preserves_length_and_order() -> None:
    client = HashEmbeddingClient(dim=8)
    texts = ["a", "b", "c"]
    out = client.embed(texts)
    assert len(out) == 3
    assert out[0].values == client.embed_one("a").values
    # Order-sensitive: swapping inputs swaps outputs.
    assert client.embed(["a", "b"])[0].values == client.embed(["b", "a"])[1].values


def test_embed_empty_batch_returns_empty() -> None:
    assert HashEmbeddingClient(dim=4).embed([]) == []


# --- DTO dimension validation + immutability -------------------------------
def test_embedding_vector_rejects_length_dim_mismatch() -> None:
    with pytest.raises(ValidationError):
        EmbeddingVector(values=(0.1, 0.2), dim=3, model_id="x")


def test_dtos_are_frozen() -> None:
    vector = HashEmbeddingClient(dim=4).embed_one("x")
    with pytest.raises(ValidationError):
        vector.dim = 8


# --- Idempotent upsert -----------------------------------------------------
def test_upsert_is_idempotent_and_overwrites() -> None:
    index = InMemoryVectorIndex(dim=4)
    client = HashEmbeddingClient(dim=4)
    index.upsert([VectorRecord(chunk_id="c1", vector=client.embed_one("alpha"), metadata={"framework": "RICE"})])
    index.upsert([VectorRecord(chunk_id="c1", vector=client.embed_one("beta"), metadata={"framework": "JTBD"})])

    assert len(index) == 1
    snap = index.snapshot()
    assert snap[0].chunk_id == "c1"
    assert snap[0].metadata["framework"] == "JTBD"  # latest write wins


# --- Query ordering --------------------------------------------------------
def test_query_orders_by_score_then_chunk_id() -> None:
    index = InMemoryVectorIndex(dim=4)
    client = HashEmbeddingClient(dim=4)
    exact = client.embed_one("alpha")
    index.upsert([
        VectorRecord(chunk_id="b", vector=exact, metadata={}),
        VectorRecord(chunk_id="a", vector=exact, metadata={}),  # identical vector -> ties with b
        VectorRecord(chunk_id="z", vector=client.embed_one("faraway"), metadata={}),
    ])

    matches = index.query(exact, top_k=3)
    assert [m.chunk_id for m in matches] == ["a", "b", "z"]  # tie a<b by id, then z
    assert matches[0].score >= matches[1].score >= matches[2].score


def test_query_respects_top_k() -> None:
    index = InMemoryVectorIndex(dim=4)
    client = HashEmbeddingClient(dim=4)
    index.upsert([VectorRecord(chunk_id=f"c{i}", vector=client.embed_one(str(i)), metadata={}) for i in range(5)])
    assert len(index.query(client.embed_one("0"), top_k=2)) == 2
    assert index.query(client.embed_one("0"), top_k=0) == []


# --- Metadata filtering ----------------------------------------------------
def test_query_filters_by_metadata() -> None:
    index = InMemoryVectorIndex(dim=4)
    client = HashEmbeddingClient(dim=4)
    index.upsert([
        VectorRecord(chunk_id="r", vector=client.embed_one("r"), metadata={"framework": "RICE"}),
        VectorRecord(chunk_id="j", vector=client.embed_one("j"), metadata={"framework": "JTBD"}),
    ])
    out = index.query(client.embed_one("q"), top_k=10, where=VectorFilter(equals={"framework": "RICE"}))
    assert [m.chunk_id for m in out] == ["r"]


# --- Delete ----------------------------------------------------------------
def test_delete_removes_records_and_is_lenient() -> None:
    index = InMemoryVectorIndex(dim=4)
    client = HashEmbeddingClient(dim=4)
    index.upsert([VectorRecord(chunk_id="x", vector=client.embed_one("x"), metadata={})])
    index.delete(["x", "missing"])  # missing id ignored
    assert len(index) == 0 and "x" not in index


# --- Snapshot --------------------------------------------------------------
def test_snapshot_preserves_insertion_order() -> None:
    index = InMemoryVectorIndex(dim=4)
    client = HashEmbeddingClient(dim=4)
    for chunk_id in ["c2", "c0", "c1"]:
        index.upsert([VectorRecord(chunk_id=chunk_id, vector=client.embed_one(chunk_id), metadata={})])
    assert [record.chunk_id for record in index.snapshot()] == ["c2", "c0", "c1"]


# --- Index dimension validation --------------------------------------------
def test_index_rejects_wrong_dim_on_upsert() -> None:
    index = InMemoryVectorIndex(dim=4)
    eight = HashEmbeddingClient(dim=8)
    with pytest.raises(EmbeddingDimensionMismatchError):
        index.upsert([VectorRecord(chunk_id="x", vector=eight.embed_one("x"), metadata={})])


def test_index_rejects_wrong_dim_on_query() -> None:
    index = InMemoryVectorIndex(dim=4)
    eight = HashEmbeddingClient(dim=8)
    with pytest.raises(EmbeddingDimensionMismatchError):
        index.query(eight.embed_one("x"), top_k=1)


# --- Offline guarantee -----------------------------------------------------
def test_vector_package_does_not_import_chromadb() -> None:
    import app.vector  # noqa: F401  -- ensure the package is imported

    assert "chromadb" not in sys.modules
