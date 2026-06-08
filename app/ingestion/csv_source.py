"""CSV ingestion -- one data row per feedback signal."""

from __future__ import annotations

import csv
import io

from app.ai_contracts.enums import StakeholderType
from app.ingestion.contract import IngestionFormat, IngestionResult, IngestionWarning, WarningKind
from app.ingestion.errors import EmptyBatchError, MissingColumnError
from app.ingestion.validators import assemble_result, decode_bytes

__all__ = ["ingest_csv"]


def ingest_csv(
    data: "bytes | str",
    *,
    text_column: str = "feedback",
    stakeholder_column: str = "stakeholder",
    default_stakeholder: StakeholderType | None = None,
    coerce_unknown: bool = False,
    dedupe: bool = True,
    delimiter: str = ",",
    filename: str = "upload.csv",
) -> IngestionResult:
    """Parse a CSV with a header row into validated entries.

    Requires ``text_column`` in the header (else :class:`MissingColumnError`). The
    ``stakeholder_column`` is optional -- absent or blank cells fall back to
    ``default_stakeholder``. ``source_ref`` is ``"{filename}:row N"`` (header is row 1).
    """

    text, decode_note = decode_bytes(data)
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    if reader.fieldnames is None:
        raise EmptyBatchError("CSV has no header row")

    fields = {(name or "").strip() for name in reader.fieldnames}
    if text_column not in fields:
        raise MissingColumnError(
            f"required text column {text_column!r} not found; header has {sorted(fields)}")
    has_stakeholder = stakeholder_column in fields

    rows = []
    for line_no, record in enumerate(reader, start=2):  # header occupies line 1
        cell = {(k or "").strip(): v for k, v in record.items()}
        raw_text = cell.get(text_column) or ""
        stakeholder = cell.get(stakeholder_column) if has_stakeholder else None
        rows.append((f"{filename}:row {line_no}", raw_text, stakeholder))

    extra = ([IngestionWarning(filename, WarningKind.DECODE_FALLBACK, decode_note)] if decode_note else [])
    return assemble_result(
        rows,
        source_format=IngestionFormat.CSV,
        source_ref=filename,
        default_stakeholder=default_stakeholder,
        coerce_unknown=coerce_unknown,
        dedupe=dedupe,
        extra_warnings=extra,
    )
