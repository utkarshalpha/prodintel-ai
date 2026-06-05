"""create knowledge_source and knowledge_chunk (RAG corpus foundations)

Revision ID: 0006_knowledge_source_and_chunk
Revises: 0005_decision_and_edges
Create Date: 2026-06-05

Mirrors app.models.knowledge exactly:

* ``knowledge_source`` -- ingested external knowledge artifact; immutable at the ORM
                          layer. UNIQUE(corpus_version, content_hash) for idempotent
                          re-ingest; indexed on corpus_version and source_type.
* ``knowledge_chunk``  -- retrievable span of a source; FK -> knowledge_source
                          ON DELETE CASCADE; UNIQUE(source_id, content_hash) for
                          per-source dedup; indexed on source_id, chroma_id,
                          corpus_version.

Enum notes
----------
The two ``*_framework_name`` enums materialize :class:`app.ai_contracts.enums.FrameworkName`
for the **first** time (it was reserved but never previously created). Their labels
use the **canonical mixed-case** framework spellings -- ``RICE``, ``JTBD``,
``MoSCoW``, ``STRATEGY``, ``PRD_TEMPLATE``, ``Kano`` -- and MUST stay byte-identical
to the enum members; ``tests/test_migrations.py`` asserts this so a lowercase slip
(which Postgres would reject at insert time but the SQLite parity check cannot see)
fails CI here. Distinct type names avoid collisions with each other and with enums
from earlier revisions.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0006_knowledge_source_and_chunk"
down_revision: Union[str, None] = "0005_decision_and_edges"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Mixed-case framework labels are intentional -- they match FrameworkName.value verbatim.
_FRAMEWORK_LABELS = ("RICE", "JTBD", "MoSCoW", "STRATEGY", "PRD_TEMPLATE", "Kano")

_knowledge_source_type = sa.Enum(
    "framework", "book", "historical_decision", "company_strategy", name="knowledge_source_type"
)
_knowledge_source_framework = sa.Enum(*_FRAMEWORK_LABELS, name="knowledge_source_framework_name")
_knowledge_chunk_framework = sa.Enum(*_FRAMEWORK_LABELS, name="knowledge_chunk_framework_name")


def upgrade() -> None:
    op.create_table(
        "knowledge_source",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=True),
        sa.Column("source_type", _knowledge_source_type, nullable=False),
        sa.Column("framework", _knowledge_source_framework, nullable=True),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("author", sa.String(length=255), nullable=True),
        sa.Column("source_license", sa.String(length=128), nullable=True),
        sa.Column("uri", sa.String(length=2048), nullable=True),
        sa.Column("corpus_version", sa.String(length=64), nullable=False),
        sa.Column("embedding_model_id", sa.String(length=128), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_knowledge_source")),
        sa.UniqueConstraint(
            "corpus_version",
            "content_hash",
            name=op.f("uq_knowledge_source_corpus_version_content_hash"),
        ),
    )
    op.create_index(
        op.f("ix_knowledge_source_corpus_version"), "knowledge_source", ["corpus_version"], unique=False
    )
    op.create_index(
        op.f("ix_knowledge_source_source_type"), "knowledge_source", ["source_type"], unique=False
    )

    op.create_table(
        "knowledge_chunk",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("source_id", sa.Uuid(), nullable=False),
        sa.Column("corpus_version", sa.String(length=64), nullable=False),
        sa.Column("embedding_model_id", sa.String(length=128), nullable=False),
        sa.Column("framework", _knowledge_chunk_framework, nullable=True),
        sa.Column("section", sa.String(length=255), nullable=True),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("chroma_id", sa.Uuid(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_knowledge_chunk")),
        sa.ForeignKeyConstraint(
            ["source_id"],
            ["knowledge_source.id"],
            name=op.f("fk_knowledge_chunk_source_id_knowledge_source"),
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "source_id", "content_hash", name=op.f("uq_knowledge_chunk_source_id_content_hash")
        ),
    )
    op.create_index(
        op.f("ix_knowledge_chunk_source_id"), "knowledge_chunk", ["source_id"], unique=False
    )
    op.create_index(
        op.f("ix_knowledge_chunk_chroma_id"), "knowledge_chunk", ["chroma_id"], unique=False
    )
    op.create_index(
        op.f("ix_knowledge_chunk_corpus_version"), "knowledge_chunk", ["corpus_version"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_knowledge_chunk_corpus_version"), table_name="knowledge_chunk")
    op.drop_index(op.f("ix_knowledge_chunk_chroma_id"), table_name="knowledge_chunk")
    op.drop_index(op.f("ix_knowledge_chunk_source_id"), table_name="knowledge_chunk")
    op.drop_table("knowledge_chunk")
    op.drop_index(op.f("ix_knowledge_source_source_type"), table_name="knowledge_source")
    op.drop_index(op.f("ix_knowledge_source_corpus_version"), table_name="knowledge_source")
    op.drop_table("knowledge_source")

    bind = op.get_bind()
    for enum_type in (_knowledge_chunk_framework, _knowledge_source_framework, _knowledge_source_type):
        enum_type.drop(bind, checkfirst=True)
