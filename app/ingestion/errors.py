"""Source-level ingestion errors.

Row-level validation failures are *collected* into ``IngestionResult.rejected`` and never
raised. These exceptions signal a fundamentally unprocessable *source* (the whole file/batch
cannot yield entries), mirroring how ``PipelineService`` reports partial progress but still
raises on truly unrecoverable conditions.
"""

from __future__ import annotations

__all__ = [
    "IngestionError",
    "UnsupportedFormatError",
    "FileDecodeError",
    "FileTooLargeError",
    "MissingColumnError",
    "CorruptDocumentError",
    "MissingDependencyError",
    "EmptyBatchError",
]


class IngestionError(Exception):
    """Base class for all source-level ingestion failures."""


class UnsupportedFormatError(IngestionError):
    """The requested format / file extension is not supported."""


class FileDecodeError(IngestionError):
    """The uploaded bytes could not be decoded as text in any attempted encoding."""


class FileTooLargeError(IngestionError):
    """The uploaded file exceeds ``MAX_FILE_SIZE_BYTES``."""


class MissingColumnError(IngestionError):
    """A required column (e.g. the text column) is absent from a CSV header."""


class CorruptDocumentError(IngestionError):
    """A PDF/DOCX could not be opened, or yielded no extractable text."""


class MissingDependencyError(IngestionError):
    """An optional parser dependency (pypdf / python-docx) is not installed."""


class EmptyBatchError(IngestionError):
    """The source produced zero rows to validate (e.g. blank file, header-only CSV)."""
