"""SQLAlchemy models for Signal and ParsedSignal (PART 4 -- database mapping).

Faithful to the finalized schema, scoped to what Stage 1 needs:

* ``Signal`` is the immutable provenance root. An ORM event blocks updates,
  mirroring the database-level immutability trigger from the schema spec.
* ``ParsedSignal`` holds the persisted, grounded Stage 1 output, one row per signal
  (``signal_id`` is unique). It knows how to build itself from a
  :class:`ParsedSignalContract`, so the service never hand-maps fields.

Two columns from the full schema -- ``workspace_id`` and ``chroma_id`` -- are present
but nullable: they are reserved for the Workspace and RAG/embedding phases and are
not populated by this Stage 1 slice. They are kept on the model so the table shape
matches the finalized design rather than being retrofitted later.

Vectors live in ChromaDB, not Postgres (``chroma_id`` is only a handle), consistent
with the architecture.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Enum as SAEnum,
    Float,
    ForeignKey,
    JSON,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    event,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.ai_contracts.enums import StakeholderType
from app.ai_contracts.stage1_signal import ParsedSignalContract
from app.db.base import Base

__all__ = ["Signal", "ParsedSignal", "SignalImmutableError"]


class SignalImmutableError(Exception):
    """Raised when code attempts to update an immutable :class:`Signal` row."""


def _utcnow() -> datetime:
    """Timezone-aware current timestamp (client-side default)."""

    return datetime.now(timezone.utc)


def _stakeholder_enum(name: str) -> SAEnum:
    """Build a portable enum column that stores the lowercase *values*.

    Using ``values_callable`` makes the stored strings match the contract/API
    values (e.g. ``"customer"``) rather than the enum member names.
    """

    return SAEnum(
        StakeholderType,
        name=name,
        values_callable=lambda enum_cls: [member.value for member in enum_cls],
    )


class Signal(Base):
    """Immutable stakeholder signal -- the root of all provenance."""

    __tablename__ = "signal"
    __table_args__ = (UniqueConstraint("content_hash", name="uq_signal_content_hash"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    # Reserved for the Workspace phase: nullable, no FK yet (no workspace table).
    workspace_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, nullable=True)
    source_type: Mapped[StakeholderType] = mapped_column(_stakeholder_enum("source_type"), nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    source_ref: Mapped[str | None] = mapped_column(String(255), nullable=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    # Reserved for the RAG/embedding phase: the ChromaDB vector handle.
    chroma_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow, nullable=False)

    parsed: Mapped["ParsedSignal | None"] = relationship(
        back_populates="signal",
        uselist=False,
        cascade="all, delete-orphan",
    )


class ParsedSignal(Base):
    """Persisted, grounded Stage 1 analysis -- one row per signal."""

    __tablename__ = "parsed_signal"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    signal_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("signal.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    intent: Mapped[str] = mapped_column(Text, nullable=False)
    stakeholder_type: Mapped[StakeholderType] = mapped_column(
        _stakeholder_enum("stakeholder_type"), nullable=False
    )
    urgency: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    sentiment: Mapped[float] = mapped_column(Float, nullable=False)
    # Grounded claims as stored JSON: [{text, source_span:[s,e], claim_confidence}].
    extracted_claims: Mapped[list] = mapped_column(JSON, nullable=False)
    # Denormalized headline score for querying/sorting, plus the full block.
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[dict] = mapped_column(JSON, nullable=False)
    model_meta: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=_utcnow, nullable=False)

    signal: Mapped[Signal] = relationship(back_populates="parsed")

    @classmethod
    def from_contract(cls, contract: ParsedSignalContract) -> "ParsedSignal":
        """Build a persistable row from a validated, grounded Stage 1 contract.

        Uses ``mode="json"`` dumps so tuples/enums become JSON-safe primitives in
        the JSON columns.
        """

        return cls(
            signal_id=contract.signal_id,
            intent=contract.intent,
            stakeholder_type=contract.stakeholder_type,
            urgency=contract.urgency,
            sentiment=contract.sentiment,
            extracted_claims=[claim.model_dump(mode="json") for claim in contract.extracted_claims],
            confidence_score=contract.confidence.score,
            confidence=contract.confidence.model_dump(mode="json"),
            model_meta=contract.model_meta.model_dump(mode="json"),
        )


@event.listens_for(Signal, "before_update", propagate=True)
def _block_signal_update(_mapper, _connection, _target: Signal) -> None:
    """Enforce signal immutability at the ORM layer (mirrors the DB trigger)."""

    raise SignalImmutableError("Signal rows are immutable and cannot be updated")
