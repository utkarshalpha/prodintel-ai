"""Migration tests: the initial revision must build a schema matching the models.

Runs ``alembic upgrade head`` against a throwaway SQLite database, introspects the
result, and asserts it matches what ``Base.metadata.create_all`` would produce --
proving the migration and the models agree. Also checks that ``downgrade base``
removes the tables.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

import app.models  # noqa: F401  -- register tables on Base.metadata
from app.ai_contracts.enums import FrameworkName
from app.db.base import Base

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ALEMBIC_DIR = _REPO_ROOT / "alembic"


def _alembic_config(url: str) -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(_ALEMBIC_DIR))
    cfg.set_main_option("sqlalchemy.url", url)
    return cfg


@pytest.fixture
def sqlite_url(tmp_path, monkeypatch) -> str:
    db_path = tmp_path / "migration_test.db"
    url = f"sqlite+pysqlite:///{db_path}"
    # env.py prefers DATABASE_URL; point it at the throwaway database.
    monkeypatch.setenv("DATABASE_URL", url)
    return url


_ALL_TABLES = (
    "signal",
    "parsed_signal",
    "feature",
    "feature_signal",
    "conflict",
    "conflict_party",
    "decision",
    "decision_evidence",
    "decision_conflict",
    "knowledge_source",
    "knowledge_chunk",
)


def _load_revision_module(filename: str):
    """Import an Alembic revision file by path (names start with a digit)."""

    path = _ALEMBIC_DIR / "versions" / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _columns(engine, table: str) -> set[str]:
    return {col["name"] for col in inspect(engine).get_columns(table)}


def _table_fingerprint(engine, table: str) -> dict:
    """Capture columns (type+nullability), PK, uniques, FKs (with ondelete), and
    explicit (non-auto) indexes for a table -- enough to detect any drift between
    the migrated schema and the model schema."""

    insp = inspect(engine)
    columns = {
        col["name"]: (str(col["type"]), bool(col["nullable"]))
        for col in insp.get_columns(table)
    }
    pk = tuple(insp.get_pk_constraint(table)["constrained_columns"])
    uniques = {frozenset(uc["column_names"]) for uc in insp.get_unique_constraints(table)}
    fks = {
        (
            tuple(fk["constrained_columns"]),
            fk["referred_table"],
            tuple(fk["referred_columns"]),
            (fk.get("options") or {}).get("ondelete"),
        )
        for fk in insp.get_foreign_keys(table)
    }
    indexes = {
        (idx["name"], tuple(idx["column_names"]))
        for idx in insp.get_indexes(table)
        if idx.get("name") and not str(idx["name"]).startswith("sqlite_autoindex")
    }
    return {"columns": columns, "pk": pk, "uniques": uniques, "fks": fks, "indexes": indexes}


def test_upgrade_creates_expected_tables(sqlite_url: str) -> None:
    command.upgrade(_alembic_config(sqlite_url), "head")

    engine = create_engine(sqlite_url, future=True)
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    assert set(_ALL_TABLES).issubset(tables)
    engine.dispose()


def test_migration_schema_matches_models(sqlite_url: str) -> None:
    """Full parity: columns/types/nullability, PK, uniques, FKs (ondelete), indexes."""

    command.upgrade(_alembic_config(sqlite_url), "head")
    migrated_engine = create_engine(sqlite_url, future=True)

    model_engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(model_engine)

    for table in _ALL_TABLES:
        assert _table_fingerprint(migrated_engine, table) == _table_fingerprint(model_engine, table), table

    migrated_engine.dispose()
    model_engine.dispose()


def test_migration_creates_reverse_provenance_index(sqlite_url: str) -> None:
    """HIGH #1: the ix_feature_signal_signal_id index is created by the migrations."""

    command.upgrade(_alembic_config(sqlite_url), "head")
    engine = create_engine(sqlite_url, future=True)
    index_names = {idx["name"] for idx in inspect(engine).get_indexes("feature_signal")}
    assert "ix_feature_signal_signal_id" in index_names
    engine.dispose()


def test_migration_creates_decision_reverse_provenance_index(sqlite_url: str) -> None:
    """The ix_decision_evidence_signal_id index is created by the migrations."""

    command.upgrade(_alembic_config(sqlite_url), "head")
    engine = create_engine(sqlite_url, future=True)
    index_names = {idx["name"] for idx in inspect(engine).get_indexes("decision_evidence")}
    assert "ix_decision_evidence_signal_id" in index_names
    engine.dispose()


def test_migration_creates_knowledge_chunk_indexes(sqlite_url: str) -> None:
    """The knowledge_chunk lookup indexes (source_id, chroma_id, corpus_version) exist."""

    command.upgrade(_alembic_config(sqlite_url), "head")
    engine = create_engine(sqlite_url, future=True)
    index_names = {idx["name"] for idx in inspect(engine).get_indexes("knowledge_chunk")}
    assert {
        "ix_knowledge_chunk_source_id",
        "ix_knowledge_chunk_chroma_id",
        "ix_knowledge_chunk_corpus_version",
    } <= index_names
    engine.dispose()


def test_migration_0006_framework_enum_labels_match_python_enum() -> None:
    """HIGH guard: the 0006 framework-enum labels equal FrameworkName.value verbatim.

    The SQLite parity check cannot see enum labels, so a lowercase slip in the
    migration (which Postgres would reject at insert time) would pass everything
    else. This asserts the migration's hardcoded labels -- including the mixed-case
    spellings and the newly added ``Kano`` -- match the canonical enum exactly.
    """

    module = _load_revision_module("0006_create_knowledge_source_and_chunk.py")
    expected = {member.value for member in FrameworkName}
    assert set(module._knowledge_source_framework.enums) == expected
    assert set(module._knowledge_chunk_framework.enums) == expected
    # The mixed-case spellings specifically must survive (regression on the HIGH bug).
    assert "MoSCoW" in expected and "Kano" in expected


def test_downgrade_removes_tables(sqlite_url: str) -> None:
    cfg = _alembic_config(sqlite_url)
    command.upgrade(cfg, "head")
    command.downgrade(cfg, "base")

    engine = create_engine(sqlite_url, future=True)
    tables = set(inspect(engine).get_table_names())
    assert "signal" not in tables
    assert "parsed_signal" not in tables
    assert "decision" not in tables
    assert "knowledge_source" not in tables
    assert "knowledge_chunk" not in tables
    engine.dispose()
