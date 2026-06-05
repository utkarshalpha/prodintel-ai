"""Repository for Conflict (and its ConflictParty children).

Thin persistence over a session: flush-not-commit (the service owns the
transaction). Conflicts are read with their parties eagerly loaded to avoid an N+1
when serializing.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.models.conflict import Conflict

__all__ = ["ConflictRepository"]


class ConflictRepository:
    """Read/insert access to :class:`Conflict`."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, conflict: Conflict) -> Conflict:
        """Insert a conflict (and its parties via cascade) and flush."""

        self._session.add(conflict)
        self._session.flush()
        return conflict

    def get(self, conflict_id: uuid.UUID) -> Conflict | None:
        return self._session.get(Conflict, conflict_id)

    def list(
        self,
        *,
        workspace_id: uuid.UUID | None = None,
        subject_id: uuid.UUID | None = None,
        limit: int = 100,
    ) -> Sequence[Conflict]:
        """List conflicts, optionally filtered by workspace or subject (feature)."""

        stmt = select(Conflict).options(selectinload(Conflict.parties))
        if subject_id is not None:
            stmt = stmt.where(Conflict.subject_id == subject_id)
        if workspace_id is not None:
            stmt = stmt.where(Conflict.workspace_id == workspace_id)
        stmt = stmt.order_by(Conflict.severity.desc(), Conflict.created_at.desc()).limit(limit)
        return list(self._session.scalars(stmt).unique())

    def list_by_subjects(self, subject_ids: Iterable[uuid.UUID]) -> Sequence[Conflict]:
        """Return **every** conflict whose subject is in ``subject_ids`` (no limit).

        Distinct from :meth:`list` (the paginated public listing, capped by ``limit``):
        decision synthesis must consider *all* conflicts over a subject so the
        completeness invariant -- "every conflict affecting a subject must be
        acknowledged" -- cannot be silently weakened by truncation. One ``IN`` query
        over the ``ix_conflict_subject_id`` index, deterministically ordered.
        """

        subjects = list(subject_ids)
        if not subjects:
            return []
        stmt = (
            select(Conflict)
            .options(selectinload(Conflict.parties))
            .where(Conflict.subject_id.in_(subjects))
            .order_by(Conflict.subject_id, Conflict.severity.desc(), Conflict.created_at.desc())
        )
        return list(self._session.scalars(stmt).unique())
