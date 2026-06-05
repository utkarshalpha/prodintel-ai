"""create conflict and conflict_party

Revision ID: 0004_conflict_conflict_party
Revises: 0003_feature_signal_signal_id_index
Create Date: 2026-06-05

Mirrors app.models.conflict exactly:

* ``conflict``        -- detected disagreement; subject_id indexed (polymorphic, no FK);
                         type/severity/status enums; confidence + model_meta JSON.
* ``conflict_party``  -- one stakeholder position; FK -> conflict ON DELETE CASCADE;
                         evidence_signal_ids stored as JSON; conflict_id indexed.

The party stakeholder enum uses a distinct type name
(``conflict_party_stakeholder_type``) so it does not collide with the existing
``stakeholder_type`` enum created in revision 0001.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0004_conflict_conflict_party"
down_revision: Union[str, None] = "0003_feature_signal_signal_id_index"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_STAKEHOLDER_VALUES = ("customer", "sales", "engineering", "support", "leadership")
_subject_type = sa.Enum("feature", "objective", name="subject_type")
_conflict_type = sa.Enum("priority", "risk", "resource", "strategic", name="conflict_type")
_conflict_status = sa.Enum("open", "acknowledged", "resolved", name="conflict_status")
_stance = sa.Enum("advocate", "oppose", "risk_flag", "neutral", name="stance")
_party_stakeholder = sa.Enum(*_STAKEHOLDER_VALUES, name="conflict_party_stakeholder_type")


def upgrade() -> None:
    op.create_table(
        "conflict",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=True),
        sa.Column("subject_type", _subject_type, nullable=False),
        sa.Column("subject_id", sa.Uuid(), nullable=False),
        sa.Column("conflict_type", _conflict_type, nullable=False),
        sa.Column("severity", sa.SmallInteger(), nullable=False),
        sa.Column("status", _conflict_status, nullable=False),
        sa.Column("confidence_score", sa.Float(), nullable=False),
        sa.Column("confidence", sa.JSON(), nullable=False),
        sa.Column("model_meta", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_conflict")),
    )
    op.create_index(op.f("ix_conflict_subject_id"), "conflict", ["subject_id"], unique=False)

    op.create_table(
        "conflict_party",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("conflict_id", sa.Uuid(), nullable=False),
        sa.Column("stakeholder_type", _party_stakeholder, nullable=False),
        sa.Column("stance", _stance, nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("evidence_signal_ids", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_conflict_party")),
        sa.ForeignKeyConstraint(
            ["conflict_id"],
            ["conflict.id"],
            name=op.f("fk_conflict_party_conflict_id_conflict"),
            ondelete="CASCADE",
        ),
    )
    op.create_index(op.f("ix_conflict_party_conflict_id"), "conflict_party", ["conflict_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_conflict_party_conflict_id"), table_name="conflict_party")
    op.drop_table("conflict_party")
    op.drop_index(op.f("ix_conflict_subject_id"), table_name="conflict")
    op.drop_table("conflict")

    bind = op.get_bind()
    for enum_type in (_party_stakeholder, _stance, _conflict_status, _conflict_type, _subject_type):
        enum_type.drop(bind, checkfirst=True)
