"""Errors raised by the Claude adapter layer.

Two failure classes, with different dispositions in the harness:

* Transient API failures (timeout, rate limit, 5xx) are raised by the adapter as
  :class:`~app.ai_runtime.errors.LLMClientError` -- which the harness catches and
  treats as a retryable ``CLIENT_ERROR``.
* :class:`ClaudeFatalError` (auth, bad request, misconfiguration) is *not* an
  ``LLMClientError`` and is *not* caught by the harness, so it propagates and fails
  fast instead of burning retry attempts on a hopeless call.
"""

from __future__ import annotations

__all__ = ["ClaudeConfigurationError", "ClaudeFatalError"]


class ClaudeConfigurationError(Exception):
    """Raised when the client cannot be configured (e.g. missing API key)."""


class ClaudeFatalError(Exception):
    """A non-retryable API failure (auth/permission/bad-request).

    Deliberately not a subclass of ``LLMClientError`` so the harness does not catch
    it and retry; it surfaces to the caller for fail-fast handling.
    """
