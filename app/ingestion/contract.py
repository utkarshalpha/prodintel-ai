"""Canonical ingestion contract -- the boundary types between L4 and the pipeline.

This module is deliberately dependency-light: it imports only ``StakeholderType`` from
``app.ai_contracts.enums``. It defines :class:`FeedbackEntry` (the validated unit the
pipeline begins at -- relocated here from ``pipeline_service`` so ingestion never depends on
the coordinator) plus the result/diagnostic types the Live UI consumes.

Centralized limits live here and nowhere else (no magic numbers in the parsers).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.ai_contracts.enums import StakeholderType

__all__ = [
    "MAX_FILE_SIZE_BYTES",
    "MAX_ROWS",
    "MAX_TEXT_LENGTH",
    "FeedbackEntry",
    "IngestionFormat",
    "RejectReason",
    "WarningKind",
    "RejectedRow",
    "IngestionWarning",
    "IngestionResult",
]

# --- centralized limits (single source of truth) --------------------------- #
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MiB per uploaded file
MAX_ROWS = 1_000                       # max feedback rows accepted per batch
MAX_TEXT_LENGTH = 10_000               # max characters per single feedback signal


@dataclass(frozen=True)
class FeedbackEntry:
    """One unit of raw stakeholder feedback, ready for Stage 1.

    ``stakeholder_type`` is an already-validated enum -- string coercion/validation is the
    ingestion layer's job, upstream of the coordinator.
    """

    stakeholder_type: StakeholderType
    text: str
    source_ref: str | None = None


class IngestionFormat(str, Enum):
    """The supported input formats."""

    MANUAL = "manual"
    CSV = "csv"
    TXT = "txt"
    PDF = "pdf"
    DOCX = "docx"


class RejectReason(str, Enum):
    """Why a single row was rejected (collected, never raised)."""

    EMPTY_TEXT = "empty_text"
    TEXT_TOO_LONG = "text_too_long"
    UNKNOWN_STAKEHOLDER = "unknown_stakeholder"
    MISSING_STAKEHOLDER = "missing_stakeholder"
    BATCH_LIMIT = "batch_limit"


class WarningKind(str, Enum):
    """Accepted-but-flagged conditions."""

    LOW_CONTENT = "low_content"                # no content tokens -> may not ground
    DEFAULTED_STAKEHOLDER = "defaulted_stakeholder"  # unknown value coerced to default
    DUPLICATE = "duplicate"                    # exact duplicate dropped
    DECODE_FALLBACK = "decode_fallback"        # bytes decoded with a non-UTF-8 fallback


@dataclass(frozen=True)
class RejectedRow:
    """A row that failed validation, with enough context to surface in the UI."""

    locator: str
    excerpt: str
    reason: RejectReason
    detail: str | None = None


@dataclass(frozen=True)
class IngestionWarning:
    """A non-fatal flag attached to an accepted (or dropped-duplicate) row."""

    locator: str
    kind: WarningKind
    detail: str | None = None


@dataclass(frozen=True)
class IngestionResult:
    """The canonical, UI-ready output of ingestion: validated entries + diagnostics.

    ``entries`` is the only thing fed to ``PipelineService.analyze``; every element has
    already passed validation. Partial success is normal: a batch may yield both ``entries``
    and ``rejected``.
    """

    entries: tuple[FeedbackEntry, ...]
    rejected: tuple[RejectedRow, ...]
    warnings: tuple[IngestionWarning, ...]
    source_format: IngestionFormat
    source_ref: str | None
    total_rows: int

    @property
    def accepted_count(self) -> int:
        return len(self.entries)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)
