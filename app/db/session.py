"""Engine and session factory wiring.

Provides the default application engine/``SessionLocal`` and the ``get_db``
dependency. Tests construct their own engine and override ``get_db`` via FastAPI's
dependency overrides, so nothing here is hard-coded to production.

``expire_on_commit=False`` keeps ORM instances usable after ``commit()`` while the
request's session is still open, which lets the service commit and then build a
response without a redundant reload.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

__all__ = ["engine", "SessionLocal", "get_db"]

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+pysqlite:///./prodintel.db")

engine = create_engine(DATABASE_URL, future=True)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    class_=Session,
)


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a request-scoped session."""

    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
