"""Format dispatch -- route an upload to the right parser by format or file extension."""

from __future__ import annotations

import os

from app.ai_contracts.enums import StakeholderType
from app.ingestion.contract import IngestionFormat, IngestionResult
from app.ingestion.csv_source import ingest_csv
from app.ingestion.docx_source import ingest_docx
from app.ingestion.errors import UnsupportedFormatError
from app.ingestion.manual import ingest_manual
from app.ingestion.pdf_source import ingest_pdf
from app.ingestion.text_source import ingest_text

__all__ = ["ingest", "format_for_filename"]

_EXTENSIONS = {
    ".csv": IngestionFormat.CSV,
    ".txt": IngestionFormat.TXT,
    ".text": IngestionFormat.TXT,
    ".pdf": IngestionFormat.PDF,
    ".docx": IngestionFormat.DOCX,
}


def format_for_filename(filename: str) -> IngestionFormat:
    """Resolve a format from a file extension, or raise :class:`UnsupportedFormatError`."""

    ext = os.path.splitext(filename)[1].lower()
    fmt = _EXTENSIONS.get(ext)
    if fmt is None:
        raise UnsupportedFormatError(f"unsupported file extension {ext!r}")
    return fmt


def ingest(
    data,
    *,
    format: IngestionFormat | str | None = None,
    filename: str | None = None,
    default_stakeholder: StakeholderType | None = None,
    **options,
) -> IngestionResult:
    """Dispatch ``data`` to the correct parser.

    Provide an explicit ``format`` or a ``filename`` to infer it from the extension.
    File parsers receive ``filename`` (for ``source_ref``) when one is given; ``MANUAL``
    expects ``data`` to be the ``(stakeholder, text)`` row sequence.
    """

    if format is not None:
        fmt = IngestionFormat(format)
    elif filename is not None:
        fmt = format_for_filename(filename)
    else:
        raise UnsupportedFormatError("provide either 'format' or 'filename'")

    file_kw = {"filename": filename} if filename is not None else {}

    if fmt is IngestionFormat.MANUAL:
        return ingest_manual(data, default_stakeholder=default_stakeholder, **options)
    if fmt is IngestionFormat.CSV:
        return ingest_csv(data, default_stakeholder=default_stakeholder, **file_kw, **options)
    if fmt is IngestionFormat.TXT:
        return ingest_text(data, default_stakeholder=default_stakeholder, **file_kw, **options)
    if fmt is IngestionFormat.PDF:
        return ingest_pdf(data, default_stakeholder=default_stakeholder, **file_kw, **options)
    if fmt is IngestionFormat.DOCX:
        return ingest_docx(data, default_stakeholder=default_stakeholder, **file_kw, **options)
    raise UnsupportedFormatError(f"unhandled format {fmt!r}")  # pragma: no cover
