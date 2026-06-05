"""Observability: structured logging, request context, and correlation ids."""

from app.observability.logging import (
    configure_logging,
    get_logger,
    log_context,
    new_correlation_id,
)

__all__ = ["configure_logging", "get_logger", "log_context", "new_correlation_id"]
