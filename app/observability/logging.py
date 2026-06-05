"""Structured (JSON) logging and request context for ProdIntel AI.

Provides:

* :class:`JsonFormatter` -- renders each record as a single JSON line, including any
  structured fields passed via ``extra=`` and any bound context.
* contextvar-based request context (:func:`log_context`, :func:`push_context`,
  :func:`reset_context`) plus :class:`ContextFilter`, which injects the current
  context (e.g. ``correlation_id``) into every record.
* :func:`configure_logging` -- installs the JSON handler on the root logger without
  clobbering other handlers (so pytest's ``caplog`` keeps working).

Logs are made testable two ways: the structured ``extra`` fields are attached as
record attributes (captured by ``caplog`` regardless of formatter), and
:class:`JsonFormatter` / :class:`ContextFilter` are unit-testable in isolation.
Secrets are never logged (the API key is a ``SecretStr`` and is not passed to logs).
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from typing import Any, Iterator

__all__ = [
    "JsonFormatter",
    "ContextFilter",
    "configure_logging",
    "get_logger",
    "new_correlation_id",
    "log_context",
    "push_context",
    "reset_context",
    "get_context",
]

# Standard LogRecord attributes; anything else on a record is treated as a
# structured field and included in the JSON payload.
_RESERVED_ATTRS = frozenset(
    {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "taskName", "message", "asctime",
    }
)

_context: ContextVar[dict[str, Any]] = ContextVar("log_context", default={})


class JsonFormatter(logging.Formatter):
    """Format a log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class ContextFilter(logging.Filter):
    """Inject the current bound context onto every record (e.g. correlation_id)."""

    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in get_context().items():
            setattr(record, key, value)
        return True


def get_context() -> dict[str, Any]:
    """Return a copy of the currently bound logging context."""

    return dict(_context.get())


def push_context(**fields: Any) -> Token:
    """Bind ``fields`` onto the logging context; returns a token for :func:`reset_context`."""

    return _context.set({**_context.get(), **fields})


def reset_context(token: Token) -> None:
    """Restore the logging context to before the matching :func:`push_context`."""

    _context.reset(token)


@contextmanager
def log_context(**fields: Any) -> Iterator[None]:
    """Scope-bind structured fields onto the logging context for the duration."""

    token = push_context(**fields)
    try:
        yield
    finally:
        reset_context(token)


def new_correlation_id() -> str:
    """Generate a fresh correlation id (hex)."""

    return uuid.uuid4().hex


def configure_logging(level: int | str = logging.INFO, *, stream: Any = None) -> logging.Handler:
    """Install the JSON handler on the root logger (idempotent, non-clobbering).

    Removes any previously installed ProdIntel handler before adding a fresh one,
    leaving other handlers (e.g. pytest's capture handler) intact. Call once at
    application startup.
    """

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = [h for h in root.handlers if not getattr(h, "_prodintel", False)]

    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(ContextFilter())
    handler._prodintel = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    return handler


def get_logger(name: str) -> logging.Logger:
    """Return a named logger (thin wrapper over :func:`logging.getLogger`)."""

    return logging.getLogger(name)
