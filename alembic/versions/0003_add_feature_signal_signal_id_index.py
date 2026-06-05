"""add index on feature_signal.signal_id

Revision ID: 0003_feature_signal_signal_id_index
Revises: 0002_feature_feature_signal
Create Date: 2026-06-04

Adds a secondary index on ``feature_signal.signal_id`` for reverse provenance
lookups ("which features came from this signal") and to support the ON DELETE
RESTRICT integrity check on ``signal``. The composite primary key
(feature_id, signal_id) leads with feature_id and cannot serve a signal_id-only
filter.
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "0003_feature_signal_signal_id_index"
down_revision: Union[str, None] = "0002_feature_feature_signal"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        op.f("ix_feature_signal_signal_id"),
        "feature_signal",
        ["signal_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_feature_signal_signal_id"), table_name="feature_signal")
