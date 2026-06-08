"""PDF ingestion (optional dependency: ``pypdf``).

Extracts a text layer per page and segments it like a text file. Scanned/image-only PDFs
have no text layer and are rejected (no OCR).
"""

from __future__ import annotations

import io

from app.ai_contracts.enums import StakeholderType
from app.ingestion.contract import IngestionFormat, IngestionResult
from app.ingestion.errors import CorruptDocumentError, MissingDependencyError
from app.ingestion.text_source import segment
from app.ingestion.validators import assemble_result, enforce_size

__all__ = ["ingest_pdf"]


def ingest_pdf(
    data: bytes,
    *,
    default_stakeholder: StakeholderType,
    granularity: str = "paragraph",
    dedupe: bool = True,
    filename: str = "upload.pdf",
) -> IngestionResult:
    """Extract text from a PDF and segment it (``source_ref`` = ``"{filename}:pN:¶M"``)."""

    enforce_size(data)
    try:
        import pypdf  # lazy: optional dependency
    except ImportError as exc:
        raise MissingDependencyError(
            "PDF ingestion requires 'pypdf' (pip install prodintel-ai[ingest])") from exc

    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        pages = [(page_no, page.extract_text() or "")
                 for page_no, page in enumerate(reader.pages, start=1)]
    except Exception as exc:  # pypdf raises a variety of read errors
        raise CorruptDocumentError(f"could not read PDF: {exc}") from exc

    rows = []
    for page_no, page_text in pages:
        for n, seg in enumerate(segment(page_text, granularity), start=1):
            rows.append((f"{filename}:p{page_no}:¶{n}", seg, None))

    if not rows:
        raise CorruptDocumentError("no extractable text (scanned or image-only PDF?)")

    return assemble_result(
        rows,
        source_format=IngestionFormat.PDF,
        source_ref=filename,
        default_stakeholder=default_stakeholder,
        coerce_unknown=False,
        dedupe=dedupe,
    )
