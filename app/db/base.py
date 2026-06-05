"""SQLAlchemy declarative base with a stable naming convention.

The naming convention gives every constraint a deterministic name, which keeps
Alembic autogenerate diffs clean and makes constraints addressable in migrations.
All ORM models inherit from :class:`Base`.
"""

from __future__ import annotations

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

__all__ = ["Base"]

# Deterministic constraint naming (index, unique, check, fk, pk).
_NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Declarative base shared by every ProdIntel AI ORM model."""

    metadata = MetaData(naming_convention=_NAMING_CONVENTION)
