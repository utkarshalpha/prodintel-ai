"""DOCX ingestion (optional dependency: ``python-docx``).

One non-empty paragraph = one feedback signal (``source_ref`` = ``"{filename}:¶N"``).
"""

from __future__ import annotations

import io

from app.ai_contracts.enums import StakeholderType
from app.ingestion.contract import IngestionFormat, IngestionResult
from app.ingestion.errors import CorruptDocumentError, EmptyBatchError, MissingDependencyError
from app.ingestion.validators import assemble_result, enforce_size

__all__ = ["ingest_docx"]


def ingest_docx(
    data: bytes,
    *,
    default_stakeholder: StakeholderType,
    dedupe: bool = True,
    filename: str = "upload.docx",
) -> IngestionResult:
    """Extract non-empty paragraphs from a .docx and validate them."""

    enforce_size(data)
    try:
        import docx  # lazy: optional dependency (python-docx)
    except ImportError as exc:
        raise MissingDependencyError(
            "DOCX ingestion requires 'python-docx' (pip install prodintel-ai[ingest])") from exc

    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:
        raise CorruptDocumentError(f"could not read DOCX: {exc}") from exc

    rows = []
    n = 0
    for paragraph in document.paragraphs:
        text = (paragraph.text or "").strip()
        if not text:
            continue
        n += 1
        rows.append((f"{filename}:¶{n}", text, None))

    if not rows:
        raise EmptyBatchError("DOCX contained no non-empty paragraphs")

    return assemble_result(
        rows,
        source_format=IngestionFormat.DOCX,
        source_ref=filename,
        default_stakeholder=default_stakeholder,
        coerce_unknown=False,
        dedupe=dedupe,
    )
