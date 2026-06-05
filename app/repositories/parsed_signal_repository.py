"""ParsedSignalRepository -- persistence for Stage 1 analysis rows.

One :class:`ParsedSignal` per signal (``signal_id`` is unique). Re-analyzing a
signal replaces its prior analysis, so :meth:`upsert_for_signal` deletes any
existing row before inserting -- keeping the unique constraint satisfied and the
latest grounded analysis authoritative. Like the signal repository, it flushes but
does not commit.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.signal import ParsedSignal

__all__ = ["ParsedSignalRepository"]


class ParsedSignalRepository:
    """Read/upsert access to :class:`ParsedSignal`."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get_by_signal_id(self, signal_id: uuid.UUID) -> ParsedSignal | None:
        """Fetch the analysis for a signal, or ``None`` if not yet analyzed."""

        return self._session.scalar(
            select(ParsedSignal).where(ParsedSignal.signal_id == signal_id)
        )

    def add(self, parsed: ParsedSignal) -> ParsedSignal:
        """Insert a new analysis row and flush."""

        self._session.add(parsed)
        self._session.flush()
        return parsed

    def upsert_for_signal(self, parsed: ParsedSignal) -> ParsedSignal:
        """Insert ``parsed``, replacing any existing analysis for the same signal."""

        existing = self.get_by_signal_id(parsed.signal_id)
        if existing is not None:
            self._session.delete(existing)
            self._session.flush()
        return self.add(parsed)
