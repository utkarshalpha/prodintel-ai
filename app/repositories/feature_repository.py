"""Repositories for Feature and FeatureSignal.

Thin persistence over a session: no business logic, flush-not-commit (the service
owns the transaction). ``FeatureRepository.create_with_signals`` builds a feature
together with its provenance edges in one step.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.ai_contracts.enums import Relationship
from app.models.feature import Feature, FeatureSignal

__all__ = ["FeatureRepository", "FeatureSignalRepository"]


class FeatureRepository:
    """Read/insert access to :class:`Feature`."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, feature_id: uuid.UUID) -> Feature | None:
        return self._session.get(Feature, feature_id)

    def get_with_signals(self, feature_id: uuid.UUID) -> Feature | None:
        """Fetch a feature with its provenance edges (``feature_signal``) loaded.

        Used by the Phase 5 explanation traversal to resolve a decision's subject
        feature and the signals behind it in one query. Returns ``None`` if absent.
        """

        stmt = (
            select(Feature)
            .where(Feature.id == feature_id)
            .options(selectinload(Feature.feature_signals))
        )
        return self._session.scalars(stmt).unique().one_or_none()

    def add(self, feature: Feature) -> Feature:
        self._session.add(feature)
        self._session.flush()
        return feature

    def create_with_signals(
        self,
        feature: Feature,
        signal_ids: Iterable[uuid.UUID],
        *,
        relationship: Relationship = Relationship.INFORMS,
    ) -> Feature:
        """Insert a feature and its provenance edges to the given signals."""

        for signal_id in signal_ids:
            feature.feature_signals.append(
                FeatureSignal(signal_id=signal_id, relationship_type=relationship)
            )
        self._session.add(feature)
        self._session.flush()
        return feature

    def list(
        self,
        *,
        workspace_id: uuid.UUID | None = None,
        signal_id: uuid.UUID | None = None,
        limit: int = 100,
    ) -> Sequence[Feature]:
        """List features, optionally filtered by workspace or a linked signal."""

        # Eager-load provenance edges to avoid an N+1 when serializing the list.
        stmt = select(Feature).options(selectinload(Feature.feature_signals))
        if signal_id is not None:
            stmt = stmt.join(FeatureSignal, FeatureSignal.feature_id == Feature.id).where(
                FeatureSignal.signal_id == signal_id
            )
        if workspace_id is not None:
            stmt = stmt.where(Feature.workspace_id == workspace_id)
        stmt = stmt.order_by(Feature.created_at.desc()).limit(limit)
        return list(self._session.scalars(stmt).unique())


class FeatureSignalRepository:
    """Read/insert access to :class:`FeatureSignal` provenance edges."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, feature_id: uuid.UUID, signal_id: uuid.UUID) -> FeatureSignal | None:
        return self._session.get(FeatureSignal, {"feature_id": feature_id, "signal_id": signal_id})

    def add(self, link: FeatureSignal) -> FeatureSignal:
        self._session.add(link)
        self._session.flush()
        return link
