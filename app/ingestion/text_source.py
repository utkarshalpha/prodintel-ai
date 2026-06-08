"""Plain-text (.txt) ingestion + the shared text segmenter (reused by PDF/DOCX)."""

from __future__ import annotations

import re

from app.ai_contracts.enums import StakeholderType
from app.ingestion.contract import IngestionFormat, IngestionResult, IngestionWarning, WarningKind
from app.ingestion.validators import assemble_result, decode_bytes

__all__ = ["ingest_text", "segment", "GRANULARITIES"]

GRANULARITIES = ("paragraph", "line", "document")
_PARAGRAPH_RE = re.compile(r"\n\s*\n")


def segment(text: str, granularity: str = "paragraph") -> list[str]:
    """Split text into signal units. Granularity: paragraph | line | document."""

    if granularity == "document":
        stripped = text.strip()
        return [stripped] if stripped else []
    if granularity == "line":
        return [ln.strip() for ln in text.splitlines() if ln.strip()]
    if granularity == "paragraph":
        return [p.strip() for p in _PARAGRAPH_RE.split(text) if p.strip()]
    raise ValueError(f"unknown granularity {granularity!r}; expected one of {GRANULARITIES}")


def _locator(filename: str, granularity: str, n: int) -> str:
    if granularity == "document":
        return filename
    marker = "¶" if granularity == "paragraph" else "line "
    return f"{filename}:{marker}{n}"


def ingest_text(
    data: "bytes | str",
    *,
    default_stakeholder: StakeholderType,
    granularity: str = "paragraph",
    dedupe: bool = True,
    filename: str = "upload.txt",
) -> IngestionResult:
    """Segment a text file (no embedded stakeholder -> ``default_stakeholder`` for all)."""

    text, decode_note = decode_bytes(data)
    segments = segment(text, granularity)
    rows = [(_locator(filename, granularity, n), seg, None) for n, seg in enumerate(segments, start=1)]
    extra = ([IngestionWarning(filename, WarningKind.DECODE_FALLBACK, decode_note)] if decode_note else [])
    return assemble_result(
        rows,
        source_format=IngestionFormat.TXT,
        source_ref=filename,
        default_stakeholder=default_stakeholder,
        coerce_unknown=False,
        dedupe=dedupe,
        extra_warnings=extra,
    )
