"""Pure validation + result assembly -- the heart of L4.

All parsers reduce their input to a flat list of ``(locator, raw_text, stakeholder_value)``
rows and call :func:`assemble_result`, which applies the approved validation rules and
accumulates a partial-success :class:`IngestionResult`. Deterministic: no clock, no
randomness; identical rows -> identical result.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from app.ai_contracts.enums import StakeholderType
from app.ai_contracts.validation.grounding import normalize as _content_tokens
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
from app.ingestion.errors import EmptyBatchError, FileDecodeError, FileTooLargeError

__all__ = [
    "normalize_text",
    "coerce_stakeholder",
    "enforce_size",
    "decode_bytes",
    "assemble_result",
]

_EXCERPT_LEN = 80

# A row as produced by every parser: (locator, raw text, stakeholder value or None).
Row = tuple[str, str, "str | StakeholderType | None"]


def normalize_text(raw: str) -> str:
    """Strip surrounding whitespace. (Internal whitespace is preserved.)"""

    return (raw or "").strip()


def coerce_stakeholder(value: "str | StakeholderType | None") -> StakeholderType | None:
    """Map a string/enum to a :class:`StakeholderType`, or ``None`` if unrecognized.

    Case-insensitive and whitespace-tolerant; ``None``/blank input returns ``None``.
    """

    if value is None:
        return None
    if isinstance(value, StakeholderType):
        return value  # already typed (Enum.__str__ would mangle it, so don't stringify)
    token = str(value).strip().lower()
    if not token:
        return None
    try:
        return StakeholderType(token)
    except ValueError:
        return None


def enforce_size(data: bytes) -> None:
    """Raise :class:`FileTooLargeError` if ``data`` exceeds ``MAX_FILE_SIZE_BYTES``."""

    if len(data) > MAX_FILE_SIZE_BYTES:
        raise FileTooLargeError(
            f"file is {len(data)} bytes; limit is {MAX_FILE_SIZE_BYTES} bytes"
        )


def decode_bytes(data: "bytes | str") -> tuple[str, str | None]:
    """Decode bytes to text. Returns ``(text, fallback_note)``.

    ``str`` is returned unchanged. Bytes are tried as UTF-8 (BOM-tolerant), then Latin-1
    with a note; a true decode failure raises :class:`FileDecodeError`.
    """

    if isinstance(data, str):
        return data, None
    enforce_size(data)
    try:
        return data.decode("utf-8-sig"), None
    except UnicodeDecodeError:
        try:
            return data.decode("latin-1"), "decoded as latin-1 (input was not valid UTF-8)"
        except UnicodeDecodeError as exc:  # pragma: no cover - latin-1 decodes any byte
            raise FileDecodeError("could not decode file as UTF-8 or Latin-1") from exc


def _excerpt(text: str) -> str:
    text = (text or "").strip()
    return text[:_EXCERPT_LEN]


def assemble_result(
    rows: Iterable[Row],
    *,
    source_format: IngestionFormat,
    source_ref: str | None,
    default_stakeholder: StakeholderType | None,
    coerce_unknown: bool = False,
    dedupe: bool = True,
    extra_warnings: Sequence[IngestionWarning] = (),
) -> IngestionResult:
    """Validate parser rows into a partial-success :class:`IngestionResult`.

    Rules (in order, per row): batch-limit -> empty-text -> length-cap -> stakeholder
    resolution -> duplicate -> low-content. Empty input (zero rows) is a source-level
    failure (:class:`EmptyBatchError`); an all-rejected batch is *not* (it returns with
    ``entries=()`` and populated ``rejected``).
    """

    rows = list(rows)
    if not rows:
        raise EmptyBatchError("source produced no rows to validate")

    entries: list[FeedbackEntry] = []
    rejected: list[RejectedRow] = []
    warnings: list[IngestionWarning] = list(extra_warnings)
    seen: set[tuple[str, str]] = set()

    for index, (locator, raw_text, stakeholder_value) in enumerate(rows):
        if index >= MAX_ROWS:
            rejected.append(RejectedRow(locator, _excerpt(raw_text), RejectReason.BATCH_LIMIT,
                                        f"exceeds row limit of {MAX_ROWS}"))
            continue

        text = normalize_text(raw_text)
        if not text:
            rejected.append(RejectedRow(locator, _excerpt(raw_text), RejectReason.EMPTY_TEXT))
            continue
        if len(text) > MAX_TEXT_LENGTH:
            rejected.append(RejectedRow(locator, _excerpt(text), RejectReason.TEXT_TOO_LONG,
                                        f"{len(text)} chars > limit {MAX_TEXT_LENGTH}"))
            continue

        # ---- stakeholder resolution -------------------------------------- #
        has_value = stakeholder_value is not None and str(stakeholder_value).strip() != ""
        if not has_value:
            if default_stakeholder is None:
                rejected.append(RejectedRow(locator, _excerpt(text), RejectReason.MISSING_STAKEHOLDER,
                                            "no stakeholder and no default supplied"))
                continue
            stakeholder = default_stakeholder
        else:
            coerced = coerce_stakeholder(stakeholder_value)
            if coerced is not None:
                stakeholder = coerced
            elif coerce_unknown and default_stakeholder is not None:
                stakeholder = default_stakeholder
                warnings.append(IngestionWarning(
                    locator, WarningKind.DEFAULTED_STAKEHOLDER,
                    f"unknown stakeholder {str(stakeholder_value)!r} coerced to {default_stakeholder.value}"))
            else:
                rejected.append(RejectedRow(locator, _excerpt(text), RejectReason.UNKNOWN_STAKEHOLDER,
                                            f"{str(stakeholder_value)!r}"))
                continue

        # ---- dedupe (within batch) --------------------------------------- #
        if dedupe:
            key = (stakeholder.value, text)
            if key in seen:
                warnings.append(IngestionWarning(locator, WarningKind.DUPLICATE,
                                                 "exact duplicate of an earlier row; dropped"))
                continue
            seen.add(key)

        # ---- low-content (warn only; the pipeline degrades gracefully) --- #
        if not _content_tokens(text):
            warnings.append(IngestionWarning(locator, WarningKind.LOW_CONTENT,
                                             "no content tokens; Stage 1 may not ground this signal"))

        entries.append(FeedbackEntry(stakeholder_type=stakeholder, text=text, source_ref=locator))

    return IngestionResult(
        entries=tuple(entries),
        rejected=tuple(rejected),
        warnings=tuple(warnings),
        source_format=source_format,
        source_ref=source_ref,
        total_rows=len(rows),
    )
