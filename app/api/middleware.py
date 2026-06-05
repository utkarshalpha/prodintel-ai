"""ASGI middleware: correlation id propagation and per-request logging.

A pure-ASGI middleware (not ``BaseHTTPMiddleware``) so the contextvar it sets in the
request task is reliably visible to the endpoint and all downstream logs. It reads an
inbound ``X-Request-ID`` (or generates one), binds it to the logging context, echoes
it on the response, and emits a structured ``http_request`` log with status and
duration.
"""

from __future__ import annotations

import time
from typing import Any, Awaitable, Callable

from app.observability.logging import get_logger, new_correlation_id, push_context, reset_context

__all__ = ["CorrelationIdMiddleware"]

_logger = get_logger("app.api.request")

Scope = dict[str, Any]
Receive = Callable[[], Awaitable[dict[str, Any]]]
Send = Callable[[dict[str, Any]], Awaitable[None]]


class CorrelationIdMiddleware:
    """Bind a correlation id per request and log request completion."""

    def __init__(self, app: Callable, header_name: str = "x-request-id") -> None:
        self.app = app
        self._header = header_name.lower().encode("latin-1")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        inbound = headers.get(self._header)
        correlation_id = inbound.decode("latin-1") if inbound else new_correlation_id()

        token = push_context(correlation_id=correlation_id)
        started = time.perf_counter()
        status_code: dict[str, int | None] = {"code": None}

        async def send_wrapper(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                status_code["code"] = message["status"]
                message.setdefault("headers", [])
                message["headers"].append((b"x-request-id", correlation_id.encode("latin-1")))
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration_ms = round((time.perf_counter() - started) * 1000.0, 2)
            _logger.info(
                "http_request",
                extra={
                    "event": "http_request",
                    "method": scope.get("method"),
                    "path": scope.get("path"),
                    "status_code": status_code["code"],
                    "duration_ms": duration_ms,
                },
            )
            reset_context(token)
