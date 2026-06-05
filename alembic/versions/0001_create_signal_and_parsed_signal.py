"""create signal and parsed_signal

Revision ID: 0001_signal_parsed_signal
Revises:
Create Date: 2026-06-04

Mirrors the current SQLAlchemy models (app.models.signal) exactly:

* ``signal``        -- immutable provenance root; UNIQUE(content_hash).
* ``parsed_signal`` -- one grounded Stage 1 analysis per signal;
                       UNIQUE(signal_id), FK -> signal ON DELETE CASCADE.

Notes
-----
* No server defaults: ``id`` and ``created_at`` are populated client-side by the
  ORM (uuid4 / utcnow), so the table carries no DEFAULT -- this matches the models.
* Signal immutability is enforced at the ORM layer (a ``before_update`` event), not
  by a database trigger, so no trigger is created here (changing that would change
  the schema, which is out of scope).
* Constraint names match the metadata naming convention, so a later
  ``alembic revision --autogenerate`` produces no spurious diffs.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001_signal_parsed_signal"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Enum value sets captured at this revision (lowercase values, matching the model's
# values_callable). On PostgreSQL these become native ENUM types created with the
# owning table; on SQLite they render as VARCHAR + CHECK.
_ENUM_VALUES = ("customer", "sales", "engineering", "support", "leadership")
_source_type = sa.Enum(*_ENUM_VALUES, name="source_type")
_stakeholder_type = sa.Enum(*_ENUM_VALUES, name="stakeholder_type")


def upgrade() -> None:
    op.create_table(
        "signal",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=True),
        sa.Column("source_type", _source_type, nullable=False),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("source_ref", sa.String(length=255), nullable=True),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("chroma_id", sa.Uuid(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_signal")),
        sa.UniqueConstraint("content_hash", name="uq_signal_content_hash"),
    )

    op.create_table(
        "parsed_signal",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("signal_id", sa.Uuid(), nullable=False),
        sa.Column("intent", sa.Text(), nullable=False),
        sa.Column("stakeholder_type", _stakeholder_type, nullable=False),
        sa.Column("urgency", sa.SmallInteger(), nullable=False),
        sa.Column("sentiment", sa.Float(), nullable=False),
        sa.Column("extracted_claims", sa.JSON(), nullable=False),
        sa.Column("confidence_score", sa.Float(), nullable=False),
        sa.Column("confidence", sa.JSON(), nullable=False),
        sa.Column("model_meta", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_parsed_signal")),
        sa.ForeignKeyConstraint(
            ["signal_id"],
            ["signal.id"],
            name=op.f("fk_parsed_signal_signal_id_signal"),
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("signal_id", name=op.f("uq_parsed_signal_signal_id")),
    )


def downgrade() -> None:
    op.drop_table("parsed_signal")
    op.drop_table("signal")
    # Tables created the native enum types (on PostgreSQL); drop them explicitly.
    # checkfirst keeps this a no-op on dialects without native enums (e.g. SQLite).
    bind = op.get_bind()
    _stakeholder_type.drop(bind, checkfirst=True)
    _source_type.drop(bind, checkfirst=True)
