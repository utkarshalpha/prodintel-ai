"""L4 ingestion -- turn user-provided content into validated ``FeedbackEntry`` objects.

A small, deterministic, self-contained library. It is independent of ``PipelineService``,
Streamlit, Claude, the stage runners, and snapshot assembly: it imports only stdlib,
``app.ai_contracts`` (enums + the grounding tokenizer), the optional parser libraries
(lazily), and its own modules. ``FeedbackEntry`` lives here (the validated unit the pipeline
begins at) and is re-exported by ``pipeline_service`` for backwards compatibility.

Public surface::

    from app.ingestion import (
        ingest, ingest_manual, ingest_csv, ingest_text, ingest_pdf, ingest_docx,
        FeedbackEntry, IngestionResult, RejectedRow, IngestionWarning,
        RejectReason, WarningKind, IngestionFormat, IngestionError,
    )
"""

from __future__ import annotations

from app.ingestion.contract import (
    MAX_FILE_SIZE_BYTES,
    MAX_ROWS,
    MAX_TEXT_LENGTH,
    FeedbackEntry,
    IngestionFormat,
    IngestionResult,
    IngestionWarning,
    RejectedRow,
    RejectReason,
    WarningKind,
)
from app.ingestion.csv_source import ingest_csv
from app.ingestion.dispatch import format_for_filename, ingest
from app.ingestion.docx_source import ingest_docx
from app.ingestion.errors import (
    CorruptDocumentError,
    EmptyBatchError,
    FileDecodeError,
    FileTooLargeError,
    IngestionError,
    MissingColumnError,
    MissingDependencyError,
    UnsupportedFormatError,
)
from app.ingestion.manual import ingest_manual
from app.ingestion.pdf_source import ingest_pdf
from app.ingestion.text_source import ingest_text

__all__ = [
    # config
    "MAX_FILE_SIZE_BYTES", "MAX_ROWS", "MAX_TEXT_LENGTH",
    # contract types
    "FeedbackEntry", "IngestionResult", "RejectedRow", "IngestionWarning",
    "RejectReason", "WarningKind", "IngestionFormat",
    # parsers / dispatch
    "ingest", "ingest_manual", "ingest_csv", "ingest_text", "ingest_pdf", "ingest_docx",
    "format_for_filename",
    # errors
    "IngestionError", "UnsupportedFormatError", "FileDecodeError", "FileTooLargeError",
    "MissingColumnError", "CorruptDocumentError", "MissingDependencyError", "EmptyBatchError",
]
