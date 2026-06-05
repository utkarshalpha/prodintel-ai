"""create feature and feature_signal

Revision ID: 0002_feature_feature_signal
Revises: 0001_signal_parsed_signal
Create Date: 2026-06-04

Mirrors app.models.feature exactly:

* ``feature``        -- normalized product feature; status enum; confidence + model_meta JSON.
* ``feature_signal`` -- provenance edge; composite PK (feature_id, signal_id);
                        FK -> feature ON DELETE CASCADE, FK -> signal ON DELETE RESTRICT.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002_feature_feature_signal"
down_revision: Union[str, None] = "0001_signal_parsed_signal"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_feature_status = sa.Enum("candidate", "confirmed", "rejected", "merged", name="feature_status")
_relationship = sa.Enum("supports", "contradicts", "informs", "grounds", name="relationship")


def upgrade() -> None:
    op.create_table(
        "feature",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=True),
        sa.Column("title", sa.String(length=120), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("jtbd", sa.Text(), nullable=False),
        sa.Column("status", _feature_status, nullable=False),
        sa.Column("confidence_score", sa.Float(), nullable=False),
        sa.Column("confidence", sa.JSON(), nullable=False),
        sa.Column("model_meta", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_feature")),
    )

    op.create_table(
        "feature_signal",
        sa.Column("feature_id", sa.Uuid(), nullable=False),
        sa.Column("signal_id", sa.Uuid(), nullable=False),
        sa.Column("relationship", _relationship, nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("feature_id", "signal_id", name=op.f("pk_feature_signal")),
        sa.ForeignKeyConstraint(
            ["feature_id"],
            ["feature.id"],
            name=op.f("fk_feature_signal_feature_id_feature"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["signal_id"],
            ["signal.id"],
            name=op.f("fk_feature_signal_signal_id_signal"),
            ondelete="RESTRICT",
        ),
    )


def downgrade() -> None:
    op.drop_table("feature_signal")
    op.drop_table("feature")
    bind = op.get_bind()
    _relationship.drop(bind, checkfirst=True)
    _feature_status.drop(bind, checkfirst=True)
