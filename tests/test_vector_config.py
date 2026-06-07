"""Phase 6C-3A -- VectorConfig + Chroma collection naming.

Covers the seven required areas: config parsing, environment validation, missing-field
errors, name determinism, collision resistance, Chroma naming compliance, and the
lazy-import guarantee (no chromadb).
"""

from __future__ import annotations

import ipaddress
import re
import sys

import pytest
from pydantic import ValidationError

from app.vector.config import ChromaMode, VectorBackend, VectorConfig
from app.vector.errors import ChromaConfigurationError
from app.vector.naming import MAX_COLLECTION_NAME_LENGTH, build_collection_name

# Chroma collection-name regex: 3-63 chars, alphanumeric start/end, [a-zA-Z0-9._-] body.
_CHROMA_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9]$")


def _is_chroma_compliant(name: str) -> bool:
    if not (3 <= len(name) <= MAX_COLLECTION_NAME_LENGTH):
        return False
    if not _CHROMA_NAME_RE.match(name):
        return False
    if ".." in name:
        return False
    try:
        ipaddress.ip_address(name)
        return False  # must not be a valid IP
    except ValueError:
        return True


def _chroma_env(**overrides) -> dict:
    env = {
        "VECTOR_BACKEND": "chroma",
        "CHROMA_MODE": "persistent",
        "CHROMA_PERSIST_DIR": "/data/chroma",
        "EMBEDDING_MODEL_ID": "all-MiniLM-L6-v2",
        "EMBEDDING_DIM": "384",
    }
    env.update(overrides)
    return env


# --- 1. Config parsing -----------------------------------------------------
def test_from_env_parses_full_chroma_config() -> None:
    cfg = VectorConfig.from_env(_chroma_env())
    assert cfg.backend is VectorBackend.CHROMA
    assert cfg.mode is ChromaMode.PERSISTENT
    assert cfg.persist_directory == "/data/chroma"
    assert cfg.embedding_model_id == "all-MiniLM-L6-v2"
    assert cfg.dim == 384
    assert cfg.distance == "cosine"


def test_from_env_http_mode() -> None:
    cfg = VectorConfig.from_env(
        _chroma_env(CHROMA_MODE="http", CHROMA_HOST="chroma.svc", CHROMA_PORT="8000", CHROMA_PERSIST_DIR="")
    )
    assert cfg.mode is ChromaMode.HTTP and cfg.host == "chroma.svc" and cfg.port == 8000


def test_config_is_frozen() -> None:
    cfg = VectorConfig.from_env({})
    with pytest.raises(ValidationError):
        cfg.backend = VectorBackend.CHROMA


# --- 2. Environment validation ---------------------------------------------
def test_empty_env_defaults_to_memory() -> None:
    cfg = VectorConfig.from_env({})
    assert cfg.backend is VectorBackend.MEMORY
    assert cfg.embedding_model_id is None and cfg.dim is None  # not required for memory


def test_unknown_backend_raises() -> None:
    with pytest.raises(ChromaConfigurationError):
        VectorConfig.from_env({"VECTOR_BACKEND": "pinecone"})


def test_unknown_mode_raises() -> None:
    with pytest.raises(ChromaConfigurationError):
        VectorConfig.from_env(_chroma_env(CHROMA_MODE="grpc"))


def test_non_integer_dim_raises() -> None:
    with pytest.raises(ChromaConfigurationError):
        VectorConfig.from_env(_chroma_env(EMBEDDING_DIM="big"))


def test_non_integer_port_raises() -> None:
    with pytest.raises(ChromaConfigurationError):
        VectorConfig.from_env(_chroma_env(CHROMA_MODE="http", CHROMA_HOST="h", CHROMA_PORT="x"))


# --- 3. Missing-field errors -----------------------------------------------
def test_chroma_without_model_id_raises() -> None:
    with pytest.raises(ChromaConfigurationError):
        VectorConfig.from_env(_chroma_env(EMBEDDING_MODEL_ID=""))


def test_chroma_without_dim_raises() -> None:
    env = _chroma_env()
    del env["EMBEDDING_DIM"]
    with pytest.raises(ChromaConfigurationError):
        VectorConfig.from_env(env)


def test_persistent_without_dir_raises() -> None:
    with pytest.raises(ChromaConfigurationError):
        VectorConfig.from_env(_chroma_env(CHROMA_PERSIST_DIR=""))


def test_http_without_host_or_port_raises() -> None:
    with pytest.raises(ChromaConfigurationError):
        VectorConfig.from_env(_chroma_env(CHROMA_MODE="http", CHROMA_PERSIST_DIR="", CHROMA_PORT="8000"))


def test_direct_construction_also_validates() -> None:
    with pytest.raises(ChromaConfigurationError):
        VectorConfig(backend=VectorBackend.CHROMA)  # missing model_id + dim


def test_collection_name_only_for_chroma() -> None:
    with pytest.raises(ChromaConfigurationError):
        VectorConfig.from_env({}).collection_name()  # memory backend


# --- 4. Name determinism ---------------------------------------------------
def test_name_is_deterministic() -> None:
    a = build_collection_name("prodintel_knowledge", "all-MiniLM-L6-v2", 384)
    b = build_collection_name("prodintel_knowledge", "all-MiniLM-L6-v2", 384)
    assert a == b
    # And via the config method.
    assert VectorConfig.from_env(_chroma_env()).collection_name() == build_collection_name(
        "prodintel_knowledge", "all-MiniLM-L6-v2", 384
    )


# --- 5. Collision resistance -----------------------------------------------
def test_different_model_or_dim_yields_different_name() -> None:
    base = build_collection_name("p", "model-a", 384)
    assert base != build_collection_name("p", "model-b", 384)  # different model
    assert base != build_collection_name("p", "model-a", 768)  # different dim


def test_slug_collision_is_broken_by_hash() -> None:
    # Two distinct ids that slugify identically must still differ via the hash suffix.
    n1 = build_collection_name("p", "Model/A", 384)
    n2 = build_collection_name("p", "Model_A", 384)
    assert _slug_equal("Model/A", "Model_A")  # same slug
    assert n1 != n2  # but distinct names


def _slug_equal(a: str, b: str) -> bool:
    from app.vector.naming import _slug

    return _slug(a) == _slug(b)


# --- 6. Chroma naming compliance -------------------------------------------
@pytest.mark.parametrize(
    "model_id, dim",
    [
        ("all-MiniLM-L6-v2", 384),
        ("text-embedding-3-large", 3072),
        ("sentence-transformers/all-mpnet-base-v2", 768),
        ("x", 1),
        ("A" * 200, 1536),  # pathologically long
        ("\U0001f916 weird ✓ name", 16),  # non-ascii
        ("...:::...", 8),  # all punctuation -> slug falls back
    ],
)
def test_generated_names_are_chroma_compliant(model_id: str, dim: int) -> None:
    name = build_collection_name("prodintel_knowledge", model_id, dim)
    assert _is_chroma_compliant(name), name


def test_long_prefix_still_compliant() -> None:
    name = build_collection_name("x" * 200, "all-MiniLM-L6-v2", 384)
    assert _is_chroma_compliant(name), name


# --- 7. Lazy-import guarantee ----------------------------------------------
def test_config_and_naming_do_not_import_chromadb() -> None:
    import app.vector.config  # noqa: F401
    import app.vector.naming  # noqa: F401

    assert "chromadb" not in sys.modules
