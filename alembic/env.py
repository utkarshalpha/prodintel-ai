"""Alembic migration environment for ProdIntel AI.

Resolves the database URL from ``DATABASE_URL`` (falling back to the ini), targets
``Base.metadata`` (importing models so every table is registered), and enables:

* ``compare_type`` -- so future autogenerate notices type changes, and
* batch mode on SQLite -- so ALTER-style migrations work on SQLite in tests.
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

# Importing the models registers all tables on Base.metadata.
import app.models  # noqa: F401
from app.db.base import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _database_url() -> str:
    """Prefer the DATABASE_URL env var; fall back to the ini's sqlalchemy.url."""

    return os.environ.get("DATABASE_URL") or config.get_main_option("sqlalchemy.url")


def run_migrations_offline() -> None:
    """Emit SQL without a live connection (``alembic upgrade head --sql``)."""

    url = _database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        render_as_batch=url.startswith("sqlite"),
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live connection."""

    connectable = create_engine(_database_url(), poolclass=pool.NullPool, future=True)
    with connectable.connect() as connection:
        is_sqlite = connection.dialect.name == "sqlite"
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            render_as_batch=is_sqlite,
        )
        with context.begin_transaction():
            context.run_migrations()
    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
