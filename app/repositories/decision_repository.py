"""Repository for Decision (and its provenance edges).

Thin persistence over a session: no business logic, flush-not-commit (the service
owns the transaction). ``create_with_links`` builds a decision together with its
evidence and acknowledged-conflict edges in one step. Decisions are listed with
their edges eagerly loaded to avoid an N+1 when serializing.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.models.decision import Decision

__all__ = ["DecisionRepository"]


class DecisionRepository:
    """Read/insert access to :class:`Decision`."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, decision: Decision) -> Decision:
        """Insert a decision (and its edges via cascade) and flush."""

        self._session.add(decision)
        self._session.flush()
        return decision

    def create_with_links(
        self,
        decision: Decision,
        signal_ids: Iterable[uuid.UUID],
        conflict_ids: Iterable[uuid.UUID],
    ) -> Decision:
        """Insert a decision and its evidence + acknowledged-conflict edges."""

        decision.attach_links(signal_ids, conflict_ids)
        self._session.add(decision)
        self._session.flush()
        return decision

    def get(self, decision_id: uuid.UUID) -> Decision | None:
        return self._session.get(Decision, decision_id)

    def get_with_provenance(self, decision_id: uuid.UUID) -> Decision | None:
        """Fetch a decision with its evidence and acknowledged-conflict edges loaded.

        Eager-loads both edge collections so the explanation traversal (Phase 5) needs
        no per-edge follow-up query. Returns ``None`` if the decision is absent.
        """

        stmt = (
            select(Decision)
            .where(Decision.id == decision_id)
            .options(
                selectinload(Decision.evidence),
                selectinload(Decision.acknowledged_conflicts),
            )
        )
        return self._session.scalars(stmt).unique().one_or_none()

    def list(
        self,
        *,
        workspace_id: uuid.UUID | None = None,
        subject_id: uuid.UUID | None = None,
        limit: int = 100,
    ) -> Sequence[Decision]:
        """List decisions, optionally filtered by workspace or subject (feature).

        Ordered by ascending priority rank (1 = highest), then most recent first.
        """

        stmt = select(Decision).options(
            selectinload(Decision.evidence),
            selectinload(Decision.acknowledged_conflicts),
        )
        if subject_id is not None:
            stmt = stmt.where(Decision.subject_id == subject_id)
        if workspace_id is not None:
            stmt = stmt.where(Decision.workspace_id == workspace_id)
        stmt = stmt.order_by(Decision.priority_rank.asc(), Decision.created_at.desc()).limit(limit)
        return list(self._session.scalars(stmt).unique())
