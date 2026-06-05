"""Enum-consistency guards across the ORM metadata (Phase 6A).

Two invariants the SQLite migration-parity check is blind to:

1. Every ``Enum`` column stores the enum *values* (not member *names*), so the
   canonical mixed-case ``FrameworkName`` spellings survive into the database. A
   regression here would only surface on Postgres at insert time.
2. Every ``Enum`` *type name* is unique across the whole schema -- two columns
   sharing a type name would collide on ``CREATE TYPE`` in Postgres.

Also pins the Phase 6 enum additions (``KANO`` on :class:`FrameworkName`, the four
:class:`KnowledgeSourceType` members).
"""

from __future__ import annotations

from sqlalchemy import Enum as SAEnum

import app.models  # noqa: F401  -- register every table on Base.metadata
from app.ai_contracts.enums import FrameworkName, KnowledgeSourceType
from app.db.base import Base


def _enum_columns():
    for table in Base.metadata.tables.values():
        for column in table.columns:
            if isinstance(column.type, SAEnum):
                yield table.name, column.name, column.type


def test_framework_name_includes_kano() -> None:
    assert FrameworkName.KANO.value == "Kano"
    # The pre-existing mixed-case members are untouched.
    assert FrameworkName.MOSCOW.value == "MoSCoW"
    assert FrameworkName.RICE.value == "RICE"


def test_knowledge_source_type_members() -> None:
    assert {m.value for m in KnowledgeSourceType} == {
        "framework",
        "book",
        "historical_decision",
        "company_strategy",
    }


def test_every_enum_column_stores_member_values() -> None:
    """Each Enum column's stored labels equal its Python enum's values verbatim."""

    for table_name, column_name, enum_type in _enum_columns():
        if enum_type.enum_class is None:
            continue  # raw-string enum (not backed by a Python Enum)
        expected = [member.value for member in enum_type.enum_class]
        assert list(enum_type.enums) == expected, f"{table_name}.{column_name}"


def test_enum_type_names_are_unique() -> None:
    """No two Enum columns share a Postgres type name (CREATE TYPE collision guard)."""

    names = [enum_type.name for _table, _column, enum_type in _enum_columns()]
    duplicates = {name for name in names if names.count(name) > 1}
    assert not duplicates, f"duplicate enum type names: {sorted(duplicates)}"


def test_knowledge_framework_enums_have_distinct_type_names() -> None:
    """The source/chunk framework enums reuse FrameworkName under distinct type names."""

    names = {
        (table, column): enum_type.name
        for table, column, enum_type in _enum_columns()
        if enum_type.enum_class is FrameworkName
    }
    assert names[("knowledge_source", "framework")] == "knowledge_source_framework_name"
    assert names[("knowledge_chunk", "framework")] == "knowledge_chunk_framework_name"
