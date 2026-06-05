"""SQLAlchemy models for Decision and its provenance edges (Stage 4 persistence).

* ``Decision`` is a synthesized, evidence-backed recommendation about a subject
  feature, with its recommendation, rank, rationale, confidence, and the model
  provenance of the run that produced it.
* ``DecisionEvidence`` is the provenance edge from a decision to a signal that
  substantiates it. ``ON DELETE CASCADE`` from the decision side and
  ``ON DELETE RESTRICT`` toward the signal, so a signal backing a decision cannot be
  deleted out from under it -- the same deletion protection ``feature_signal`` gives
  features (evidence traceability is a data-model invariant, ADR-011).
* ``DecisionConflict`` is the edge linking a decision to a conflict it acknowledges.
  ``ON DELETE CASCADE`` from the decision side and ``ON DELETE RESTRICT`` toward the
  conflict, so an acknowledged conflict cannot be deleted while a decision references
  it.

Design notes:

* ``subject_id`` is a polymorphic reference (feature | objective) and therefore is
  not a database foreign key (mirroring ``conflict.subject_id``). It is indexed for
  "the decision about this feature" lookups; subject validity is enforced by the
  integrity validator at synthesis time.
* The enum *type names* are distinct (``decision_subject_type``,
  ``decision_recommendation``, ``decision_status``) so they do not collide with the
  ``subject_type`` enum created for ``conflict`` -- the same disambiguation used for
  ``conflict_party_stakeholder_type``.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from datetime import datetime, timezone

from sqlalchemy import (
    Enum as SAEnum,
    Float,
    ForeignKey,
    Index,
    JSON,
    SmallInteger,
    String,
    Text,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.ai_contracts.enums import DecisionRecommendation, DecisionStatus, SubjectType
from app.ai_contracts.stage4_decision import DecisionContract
from app.db.base import Base

__all__ = ["Decision", "DecisionEvidence", "DecisionConflict"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _value_enum(enum_cls, name: str) -> SAEnum:
    """Portable enum column storing the lowercase enum *values*."""

    return SAEnum(enum_cls, name=name, values_callable=lambda e: [member.value for member in e])


class Decision(Base):
    """A synthesized, evidence-backed decision about a subject feature."""

    __tablename__ = "decision"
    __table_args__ = (Index("ix_decision_subject_id", "subject_id"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    workspace_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, nullable=True)
    subject_type: Mapped[SubjectType] = mapped_column(
        _value_enum(SubjectType, "decision_subject_type"), nullable=False
    )
    # Polymorphic reference (feature | objective): indexed, not a DB foreign key.
    subject_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    recommendation: Mapped[DecisionRecommendation] = mapped_column(
        _value_enum(DecisionRecommendation, "decision_recommendation"), nullable=False
    )
    title: Mapped[str] = mapped_column(String(160), nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    priority_rank: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    status: Mapped[DecisionStatus] = mapped_column(
        _value_enum(DecisionStatus, "decision_status"),
        nullable=False,
        default=DecisionStatus.PROPOSED,
    )
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[dict] = mapped_column(JSON, nullable=False)
    model_meta: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow, nullable=False)

    evidence: Mapped[list["DecisionEvidence"]] = relationship(
        back_populates="decision",
        cascade="all, delete-orphan",
    )
    acknowledged_conflicts: Mapped[list["DecisionConflict"]] = relationship(
        back_populates="decision",
        cascade="all, delete-orphan",
    )

    @classmethod
    def from_contract(
        cls,
        item: DecisionContract,
        model_meta: dict,
        *,
        workspace_id: uuid.UUID | None = None,
    ) -> "Decision":
        """Build a persistable decision (scalar fields only) from a synthesized decision.

        The per-decision ``evidence_signal_ids`` and ``acknowledged_conflict_ids`` are
        persisted separately as :class:`DecisionEvidence` / :class:`DecisionConflict`
        edges by the repository.
        """

        return cls(
            workspace_id=workspace_id,
            subject_type=item.subject_type,
            subject_id=item.subject_id,
            recommendation=item.recommendation,
            title=item.title,
            rationale=item.rationale,
            priority_rank=item.priority_rank,
            confidence_score=item.confidence.score,
            confidence=item.confidence.model_dump(mode="json"),
            model_meta=model_meta,
        )

    def attach_links(
        self,
        signal_ids: Iterable[uuid.UUID],
        conflict_ids: Iterable[uuid.UUID],
    ) -> "Decision":
        """Append the evidence and acknowledged-conflict edges (deduplicated)."""

        for signal_id in dict.fromkeys(signal_ids):
            self.evidence.append(DecisionEvidence(signal_id=signal_id))
        for conflict_id in dict.fromkeys(conflict_ids):
            self.acknowledged_conflicts.append(DecisionConflict(conflict_id=conflict_id))
        return self


class DecisionEvidence(Base):
    """Provenance edge linking a decision to a signal that substantiates it."""

    __tablename__ = "decision_evidence"
    # Secondary index for reverse provenance lookups ("which decisions used this
    # signal") and the ON DELETE RESTRICT integrity check; the composite PK leads with
    # decision_id and cannot serve a signal_id-only filter.
    __table_args__ = (Index("ix_decision_evidence_signal_id", "signal_id"),)

    decision_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("decision.id", ondelete="CASCADE"),
        primary_key=True,
    )
    signal_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("signal.id", ondelete="RESTRICT"),
        primary_key=True,
    )
    created_at: Mapped[datetime] = mapped_column(default=_utcnow, nullable=False)

    decision: Mapped[Decision] = relationship(back_populates="evidence")


class DecisionConflict(Base):
    """Edge linking a decision to a conflict it acknowledges and addresses."""

    __tablename__ = "decision_conflict"
    __table_args__ = (Index("ix_decision_conflict_conflict_id", "conflict_id"),)

    decision_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("decision.id", ondelete="CASCADE"),
        primary_key=True,
    )
    conflict_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("conflict.id", ondelete="RESTRICT"),
        primary_key=True,
    )
    created_at: Mapped[datetime] = mapped_column(default=_utcnow, nullable=False)

    decision: Mapped[Decision] = relationship(back_populates="acknowledged_conflicts")
