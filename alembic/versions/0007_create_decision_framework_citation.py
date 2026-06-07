"""create decision_framework_citation (framework-grounding provenance edge)

Revision ID: 0007_decision_framework_citation
Revises: 0006_knowledge_source_and_chunk
Create Date: 2026-06-07

Mirrors the ``DecisionFrameworkCitation`` model in app.models.decision:

* composite PK ``(decision_id, chunk_id)``;
* FK -> ``decision.id`` ON DELETE CASCADE;
* FK -> ``knowledge_chunk.id`` ON DELETE RESTRICT (a cited framework chunk cannot be
  deleted while a decision grounds on it -- ADR-011 extended to the corpus);
* ``relationship_type`` / ``evidence_type`` carrying the reserved provenance typing
  (defaults GROUNDS / FRAMEWORK_CITATION), under distinct enum type names so they do
  not collide with the ``relationship`` enum from ``feature_signal``;
* nullable ``retrieval_score``;
* reverse-lookup index ``ix_decision_framework_citation_chunk_id``.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0007_decision_framework_citation"
down_revision: Union[str, None] = "0006_knowledge_source_and_chunk"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_decision_framework_relationship = sa.Enum(
    "supports", "contradicts", "informs", "grounds", name="decision_framework_relationship"
)
_decision_framework_evidence_type = sa.Enum(
    "signal", "parsed_claim", "score", "conflict", "framework_citation",
    name="decision_framework_evidence_type",
)


def upgrade() -> None:
    op.create_table(
        "decision_framework_citation",
        sa.Column("decision_id", sa.Uuid(), nullable=False),
        sa.Column("chunk_id", sa.Uuid(), nullable=False),
        sa.Column("relationship_type", _decision_framework_relationship, nullable=False),
        sa.Column("evidence_type", _decision_framework_evidence_type, nullable=False),
        sa.Column("retrieval_score", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint(
            "decision_id", "chunk_id", name=op.f("pk_decision_framework_citation")
        ),
        sa.ForeignKeyConstraint(
            ["decision_id"],
            ["decision.id"],
            name=op.f("fk_decision_framework_citation_decision_id_decision"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["chunk_id"],
            ["knowledge_chunk.id"],
            name=op.f("fk_decision_framework_citation_chunk_id_knowledge_chunk"),
            ondelete="RESTRICT",
        ),
    )
    op.create_index(
        op.f("ix_decision_framework_citation_chunk_id"),
        "decision_framework_citation",
        ["chunk_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_decision_framework_citation_chunk_id"), table_name="decision_framework_citation"
    )
    op.drop_table("decision_framework_citation")

    bind = op.get_bind()
    for enum_type in (_decision_framework_evidence_type, _decision_framework_relationship):
        enum_type.drop(bind, checkfirst=True)
