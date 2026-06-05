"""Configuration for the Claude tool client.

Holds the model, limits, and credentials, and knows how to build itself from
environment variables. The API key is a :class:`~pydantic.SecretStr` so it never
appears in logs or ``repr`` output.

The SDK's own retry count defaults to ``0`` here on purpose: the runtime harness
owns retries deterministically, so the adapter must not add a second, hidden retry
loop underneath it.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from app.ai_clients.errors import ClaudeConfigurationError

__all__ = ["ClaudeClientConfig"]

#: Sensible default analysis model; override via ANTHROPIC_MODEL.
DEFAULT_MODEL = "claude-sonnet-4-6"


class ClaudeClientConfig(BaseModel):
    """Immutable configuration for :class:`~app.ai_clients.claude_client.ClaudeToolClient`."""

    model_config = ConfigDict(frozen=True)

    api_key: SecretStr = Field(..., description="Anthropic API key (kept secret).")
    model: str = Field(default=DEFAULT_MODEL, min_length=1, description="Model id to call.")
    max_tokens: int = Field(default=2048, ge=1, description="Max output tokens per call.")
    timeout_seconds: float = Field(default=60.0, gt=0.0, description="Per-request timeout.")
    max_retries: int = Field(default=0, ge=0, description="SDK-internal retries; 0 (harness owns retries).")
    base_url: str | None = Field(default=None, description="Optional API base URL override.")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "ClaudeClientConfig":
        """Build a config from environment variables.

        Reads ``ANTHROPIC_API_KEY`` (required), ``ANTHROPIC_MODEL``,
        ``ANTHROPIC_MAX_TOKENS``, ``ANTHROPIC_TIMEOUT_SECONDS``, and
        ``ANTHROPIC_BASE_URL``. Raises :class:`ClaudeConfigurationError` if the key
        is missing.
        """

        source = os.environ if env is None else env
        api_key = source.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ClaudeConfigurationError("ANTHROPIC_API_KEY is not set")

        return cls(
            api_key=SecretStr(api_key),
            model=source.get("ANTHROPIC_MODEL", DEFAULT_MODEL),
            max_tokens=int(source.get("ANTHROPIC_MAX_TOKENS", "2048")),
            timeout_seconds=float(source.get("ANTHROPIC_TIMEOUT_SECONDS", "60")),
            base_url=source.get("ANTHROPIC_BASE_URL") or None,
        )
