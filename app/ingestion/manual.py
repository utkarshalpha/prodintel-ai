"""Manual stakeholder entry -- structured rows straight from the UI form."""

from __future__ import annotations

from collections.abc import Sequence

from app.ai_contracts.enums import StakeholderType
from app.ingestion.contract import IngestionFormat, IngestionResult
from app.ingestion.validators import assemble_result

__all__ = ["ingest_manual"]


def ingest_manual(
    rows: Sequence[tuple["str | StakeholderType | None", str]],
    *,
    default_stakeholder: StakeholderType | None = None,
    coerce_unknown: bool = False,
    dedupe: bool = True,
) -> IngestionResult:
    """Validate ``(stakeholder, text)`` rows into an :class:`IngestionResult`.

    Each row carries its own stakeholder (the UI dropdown); ``default_stakeholder`` is only
    a fallback for blank values.
    """

    parsed = [(f"manual:{n}", text, stakeholder) for n, (stakeholder, text) in enumerate(rows, start=1)]
    return assemble_result(
        parsed,
        source_format=IngestionFormat.MANUAL,
        source_ref="manual",
        default_stakeholder=default_stakeholder,
        coerce_unknown=coerce_unknown,
        dedupe=dedupe,
    )
