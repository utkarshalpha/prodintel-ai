"""SQLAlchemy models for Feature and FeatureSignal (Stage 2 persistence).

* ``Feature`` is a normalized product feature with its confidence and the model
  provenance of the run that produced it.
* ``FeatureSignal`` is the provenance edge: a many-to-many link from a feature to
  the signals it was derived from. ``ON DELETE CASCADE`` from the feature side, and
  ``ON DELETE RESTRICT`` toward the signal so a signal that backs a feature cannot
  be deleted out from under it (provenance is protected).

``workspace_id`` is reserved/nullable, consistent with the Signal model.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Enum as SAEnum,
    Float,
    ForeignKey,
    Index,
    JSON,
    String,
    Text,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.ai_contracts.enums import FeatureStatus, Relationship
from app.ai_contracts.stage2_feature import FeatureContract
from app.db.base import Base

__all__ = ["Feature", "FeatureSignal"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _value_enum(enum_cls, name: str) -> SAEnum:
    """Portable enum column storing the lowercase enum *values*."""

    return SAEnum(enum_cls, name=name, values_callable=lambda e: [member.value for member in e])


class Feature(Base):
    """A normalized product feature derived from one or more signals."""

    __tablename__ = "feature"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    workspace_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, nullable=True)
    title: Mapped[str] = mapped_column(String(120), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    jtbd: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[FeatureStatus] = mapped_column(
        _value_enum(FeatureStatus, "feature_status"),
        nullable=False,
        default=FeatureStatus.CANDIDATE,
    )
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[dict] = mapped_column(JSON, nullable=False)
    model_meta: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow, nullable=False)

    feature_signals: Mapped[list["FeatureSignal"]] = relationship(
        back_populates="feature",
        cascade="all, delete-orphan",
    )

    @classmethod
    def from_contract(
        cls,
        item: FeatureContract,
        model_meta: dict,
        *,
        workspace_id: uuid.UUID | None = None,
    ) -> "Feature":
        """Build a persistable feature from one extracted feature item.

        ``model_meta`` is the run-level provenance (shared across the run's
        features); the per-feature ``source_signal_ids`` are persisted separately as
        :class:`FeatureSignal` links by the repository.
        """

        return cls(
            workspace_id=workspace_id,
            title=item.title,
            description=item.description,
            jtbd=item.jtbd,
            confidence_score=item.confidence.score,
            confidence=item.confidence.model_dump(mode="json"),
            model_meta=model_meta,
        )


class FeatureSignal(Base):
    """Provenance edge linking a feature to a source signal."""

    __tablename__ = "feature_signal"
    # Secondary index for reverse provenance lookups ("which features came from this
    # signal") and the ON DELETE RESTRICT integrity check; the composite PK leads
    # with feature_id and cannot serve a signal_id-only filter.
    __table_args__ = (Index("ix_feature_signal_signal_id", "signal_id"),)

    feature_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("feature.id", ondelete="CASCADE"),
        primary_key=True,
    )
    signal_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("signal.id", ondelete="RESTRICT"),
        primary_key=True,
    )
    relationship_type: Mapped[Relationship] = mapped_column(
        "relationship",
        _value_enum(Relationship, "relationship"),
        nullable=False,
        default=Relationship.INFORMS,
    )
    created_at: Mapped[datetime] = mapped_column(default=_utcnow, nullable=False)

    feature: Mapped[Feature] = relationship(back_populates="feature_signals")
