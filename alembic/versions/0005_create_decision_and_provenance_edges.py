"""create decision, decision_evidence and decision_conflict

Revision ID: 0005_decision_and_edges
Revises: 0004_conflict_conflict_party
Create Date: 2026-06-05

Mirrors app.models.decision exactly:

* ``decision``           -- synthesized decision; subject_id indexed (polymorphic, no FK);
                            recommendation/status enums; confidence + model_meta JSON.
* ``decision_evidence``  -- provenance edge to a signal; composite PK
                            (decision_id, signal_id); FK -> decision ON DELETE CASCADE,
                            FK -> signal ON DELETE RESTRICT; signal_id indexed.
* ``decision_conflict``  -- edge to an acknowledged conflict; composite PK
                            (decision_id, conflict_id); FK -> decision ON DELETE CASCADE,
                            FK -> conflict ON DELETE RESTRICT; conflict_id indexed.

The decision enums use distinct type names (``decision_subject_type``,
``decision_recommendation``, ``decision_status``) so they do not collide with the
``subject_type`` enum created in revision 0004.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0005_decision_and_edges"
down_revision: Union[str, None] = "0004_conflict_conflict_party"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_decision_subject_type = sa.Enum("feature", "objective", name="decision_subject_type")
_decision_recommendation = sa.Enum(
    "build_now", "build_later", "reject", "needs_discussion", name="decision_recommendation"
)
_decision_status = sa.Enum("proposed", "accepted", "overridden", "rejected", name="decision_status")


def upgrade() -> None:
    op.create_table(
        "decision",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=True),
        sa.Column("subject_type", _decision_subject_type, nullable=False),
        sa.Column("subject_id", sa.Uuid(), nullable=False),
        sa.Column("recommendation", _decision_recommendation, nullable=False),
        sa.Column("title", sa.String(length=160), nullable=False),
        sa.Column("rationale", sa.Text(), nullable=False),
        sa.Column("priority_rank", sa.SmallInteger(), nullable=False),
        sa.Column("status", _decision_status, nullable=False),
        sa.Column("confidence_score", sa.Float(), nullable=False),
        sa.Column("confidence", sa.JSON(), nullable=False),
        sa.Column("model_meta", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_decision")),
    )
    op.create_index(op.f("ix_decision_subject_id"), "decision", ["subject_id"], unique=False)

    op.create_table(
        "decision_evidence",
        sa.Column("decision_id", sa.Uuid(), nullable=False),
        sa.Column("signal_id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("decision_id", "signal_id", name=op.f("pk_decision_evidence")),
        sa.ForeignKeyConstraint(
            ["decision_id"],
            ["decision.id"],
            name=op.f("fk_decision_evidence_decision_id_decision"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["signal_id"],
            ["signal.id"],
            name=op.f("fk_decision_evidence_signal_id_signal"),
            ondelete="RESTRICT",
        ),
    )
    op.create_index(
        op.f("ix_decision_evidence_signal_id"), "decision_evidence", ["signal_id"], unique=False
    )

    op.create_table(
        "decision_conflict",
        sa.Column("decision_id", sa.Uuid(), nullable=False),
        sa.Column("conflict_id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("decision_id", "conflict_id", name=op.f("pk_decision_conflict")),
        sa.ForeignKeyConstraint(
            ["decision_id"],
            ["decision.id"],
            name=op.f("fk_decision_conflict_decision_id_decision"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["conflict_id"],
            ["conflict.id"],
            name=op.f("fk_decision_conflict_conflict_id_conflict"),
            ondelete="RESTRICT",
        ),
    )
    op.create_index(
        op.f("ix_decision_conflict_conflict_id"), "decision_conflict", ["conflict_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_decision_conflict_conflict_id"), table_name="decision_conflict")
    op.drop_table("decision_conflict")
    op.drop_index(op.f("ix_decision_evidence_signal_id"), table_name="decision_evidence")
    op.drop_table("decision_evidence")
    op.drop_index(op.f("ix_decision_subject_id"), table_name="decision")
    op.drop_table("decision")

    bind = op.get_bind()
    for enum_type in (_decision_status, _decision_recommendation, _decision_subject_type):
        enum_type.drop(bind, checkfirst=True)
