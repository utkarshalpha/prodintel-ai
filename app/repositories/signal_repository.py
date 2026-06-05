"""SignalRepository -- persistence for immutable Signal rows.

Thin data-access object over a SQLAlchemy :class:`Session`. It holds no business
logic: hashing, dedupe policy, and transaction boundaries live in the service. The
repository only adds and queries, and it never updates (signals are immutable).

It flushes after inserts so server/client defaults (id, created_at) are populated on
the returned instance, but it does **not** commit -- the service owns the
unit-of-work so multiple repository calls can share one transaction.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.signal import Signal

__all__ = ["SignalRepository"]


class SignalRepository:
    """Read/insert access to :class:`Signal`."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, signal: Signal) -> Signal:
        """Insert a new signal and flush so defaults are populated."""

        self._session.add(signal)
        self._session.flush()
        return signal

    def get(self, signal_id: uuid.UUID) -> Signal | None:
        """Fetch a signal by primary key, or ``None`` if absent."""

        return self._session.get(Signal, signal_id)

    def get_by_content_hash(self, content_hash: str) -> Signal | None:
        """Fetch the signal with the given content hash, or ``None`` (dedupe lookup)."""

        return self._session.scalar(select(Signal).where(Signal.content_hash == content_hash))

    def exists_by_content_hash(self, content_hash: str) -> bool:
        """Whether a signal with the given content hash already exists."""

        return self.get_by_content_hash(content_hash) is not None
