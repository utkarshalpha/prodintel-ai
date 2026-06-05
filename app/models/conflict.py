"""SQLAlchemy models for Conflict and ConflictParty (Stage 3 persistence).

* ``Conflict`` is a detected disagreement over a subject feature, with its type,
  severity, confidence, and the model provenance of the run that produced it.
* ``ConflictParty`` is one stakeholder's position in that conflict, including the
  evidence signals backing it.

Two design notes:

* ``subject_id`` is a polymorphic reference (feature | objective) and therefore is
  not a database foreign key (mirroring the architecture's polymorphic-edge
  pattern). It is indexed for "conflicts about this feature" lookups; subject
  validity is enforced by the integrity validator at detection time.
* A party's ``evidence_signal_ids`` is stored as a JSON array rather than a third
  edge table, honoring the two-table design for this stage. The ids are validated
  against the real input signals before persistence, so conflicts remain
  evidence-traceable; the trade-off is that these references are not protected by a
  database FK (unlike ``feature_signal``).
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
    SmallInteger,
    Text,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.ai_contracts.enums import (
    ConflictStatus,
    ConflictType,
    Stance,
    StakeholderType,
    SubjectType,
)
from app.ai_contracts.stage3_conflict import ConflictContract
from app.db.base import Base

__all__ = ["Conflict", "ConflictParty"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _value_enum(enum_cls, name: str) -> SAEnum:
    """Portable enum column storing the lowercase enum *values*."""

    return SAEnum(enum_cls, name=name, values_callable=lambda e: [member.value for member in e])


class Conflict(Base):
    """A detected conflict between stakeholders over a subject feature."""

    __tablename__ = "conflict"
    __table_args__ = (Index("ix_conflict_subject_id", "subject_id"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    workspace_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, nullable=True)
    subject_type: Mapped[SubjectType] = mapped_column(_value_enum(SubjectType, "subject_type"), nullable=False)
    # Polymorphic reference (feature | objective): indexed, not a DB foreign key.
    subject_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    conflict_type: Mapped[ConflictType] = mapped_column(_value_enum(ConflictType, "conflict_type"), nullable=False)
    severity: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    status: Mapped[ConflictStatus] = mapped_column(
        _value_enum(ConflictStatus, "conflict_status"),
        nullable=False,
        default=ConflictStatus.OPEN,
    )
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[dict] = mapped_column(JSON, nullable=False)
    model_meta: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow, nullable=False)

    parties: Mapped[list["ConflictParty"]] = relationship(
        back_populates="conflict",
        cascade="all, delete-orphan",
    )

    @classmethod
    def from_contract(
        cls,
        item: ConflictContract,
        model_meta: dict,
        *,
        workspace_id: uuid.UUID | None = None,
    ) -> "Conflict":
        """Build a persistable conflict (with its parties) from a detected conflict."""

        conflict = cls(
            workspace_id=workspace_id,
            subject_type=item.subject_type,
            subject_id=item.subject_id,
            conflict_type=item.conflict_type,
            severity=item.severity,
            confidence_score=item.confidence.score,
            confidence=item.confidence.model_dump(mode="json"),
            model_meta=model_meta,
        )
        for position in item.positions:
            conflict.parties.append(
                ConflictParty(
                    stakeholder_type=position.stakeholder,
                    stance=position.stance,
                    summary=position.summary,
                    evidence_signal_ids=[str(signal_id) for signal_id in position.evidence_signal_ids],
                )
            )
        return conflict


class ConflictParty(Base):
    """One stakeholder's position within a conflict, with its evidence."""

    __tablename__ = "conflict_party"
    __table_args__ = (Index("ix_conflict_party_conflict_id", "conflict_id"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    conflict_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("conflict.id", ondelete="CASCADE"),
        nullable=False,
    )
    # Distinct enum type name to avoid colliding with the parsed_signal stakeholder_type.
    stakeholder_type: Mapped[StakeholderType] = mapped_column(
        _value_enum(StakeholderType, "conflict_party_stakeholder_type"), nullable=False
    )
    stance: Mapped[Stance] = mapped_column(_value_enum(Stance, "stance"), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    # Evidence signal ids (validated against real input signals at detection time).
    evidence_signal_ids: Mapped[list] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow, nullable=False)

    conflict: Mapped[Conflict] = relationship(back_populates="parties")
