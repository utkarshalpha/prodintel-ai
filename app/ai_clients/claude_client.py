"""ClaudeToolClient -- production adapter implementing the ToolCallClient protocol.

Translates the harness's provider-neutral request (system, messages, tool) into an
Anthropic Messages API call with **forced** tool use, then normalizes the response
back into a :class:`~app.ai_runtime.interfaces.LLMToolResponse` the harness already
understands. It implements the existing protocol structurally; no existing interface
is modified.

Responsibilities handled here:

* **Forced tool usage / structured output** -- ``tool_choice={"type":"tool",...}``
  so the model must return the tool's structured input.
* **Timeout handling** -- a per-request timeout on the SDK client; timeouts surface
  as retryable :class:`LLMClientError`.
* **Retryable vs fatal errors** -- transient failures (timeout, rate limit, 5xx) ->
  ``LLMClientError`` (harness retries); permanent failures (auth, bad request) ->
  :class:`ClaudeFatalError` (fail fast).
* **Token & stop-reason collection** -- pulled from ``usage`` and ``stop_reason``.

Message adaptation: the harness may append several consecutive ``user`` turns
(initial prompt + correction messages). The Anthropic API requires alternating
roles, so consecutive same-role turns are coalesced into one before sending. This is
adapter responsibility and leaves the harness untouched.
"""

from __future__ import annotations

from typing import Any, Sequence

import anthropic

from app.ai_clients.config import ClaudeClientConfig
from app.ai_clients.errors import ClaudeFatalError
from app.ai_runtime.errors import LLMClientError
from app.ai_runtime.interfaces import LLMMessage, LLMToolResponse, ToolSpec

__all__ = ["ClaudeToolClient"]

#: HTTP statuses worth retrying. Everything else (401/403/404/400/422) is fatal.
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})


def _build_sdk_client(config: ClaudeClientConfig) -> anthropic.Anthropic:
    """Construct the underlying Anthropic SDK client from config."""

    kwargs: dict[str, Any] = {
        "api_key": config.api_key.get_secret_value(),
        "max_retries": config.max_retries,  # harness owns retries; keep SDK at 0
        "timeout": config.timeout_seconds,
    }
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return anthropic.Anthropic(**kwargs)


class ClaudeToolClient:
    """Anthropic-backed implementation of the ``ToolCallClient`` protocol."""

    def __init__(self, config: ClaudeClientConfig, *, client: anthropic.Anthropic | None = None) -> None:
        """Parameters
        ----------
        config:
            Model, limits, and credentials.
        client:
            Optional pre-built SDK client; injected in tests to avoid network. When
            omitted, one is constructed from ``config``.
        """

        self._config = config
        self._client = client if client is not None else _build_sdk_client(config)

    def complete(
        self,
        *,
        system: str,
        messages: Sequence[LLMMessage],
        tool: ToolSpec,
    ) -> LLMToolResponse:
        """Make one forced-tool call and normalize the result.

        Raises
        ------
        LLMClientError
            On transient/retryable failures (timeout, rate limit, 5xx, unknown API
            errors). The harness catches this and retries.
        ClaudeFatalError
            On permanent failures (auth, permission, bad request). Not caught by the
            harness; surfaces to the caller.
        """

        request: dict[str, Any] = {
            "model": self._config.model,
            "max_tokens": self._config.max_tokens,
            "system": system,
            "messages": self._coalesce_messages(messages),
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                }
            ],
            "tool_choice": {"type": "tool", "name": tool.name},
        }

        try:
            raw = self._client.messages.create(**request)
        except anthropic.APIConnectionError as exc:  # includes APITimeoutError
            raise LLMClientError(f"connection/timeout error: {exc}") from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code in _RETRYABLE_STATUS:
                raise LLMClientError(f"retryable API status {exc.status_code}: {exc}") from exc
            raise ClaudeFatalError(f"non-retryable API status {exc.status_code}: {exc}") from exc
        except anthropic.APIError as exc:  # any other SDK error -> assume transient
            raise LLMClientError(f"anthropic API error: {exc}") from exc

        return self._to_response(raw, tool.name)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _coalesce_messages(messages: Sequence[LLMMessage]) -> list[dict[str, str]]:
        """Merge consecutive same-role turns so the API sees alternating roles."""

        out: list[dict[str, str]] = []
        for message in messages:
            if out and out[-1]["role"] == message.role:
                out[-1]["content"] = f"{out[-1]['content']}\n\n{message.content}"
            else:
                out.append({"role": message.role, "content": message.content})
        return out

    def _to_response(self, raw: Any, expected_tool: str) -> LLMToolResponse:
        """Normalize an Anthropic message into an :class:`LLMToolResponse`.

        Extracts the forced tool's structured input, any text preamble, token usage,
        and the stop reason. If no matching tool-use block is present (e.g. truncated
        output), ``tool_name``/``tool_input`` are ``None`` and the harness classifies
        the attempt accordingly.
        """

        tool_name: str | None = None
        tool_input: dict[str, Any] | None = None
        texts: list[str] = []

        for block in getattr(raw, "content", None) or []:
            block_type = getattr(block, "type", None)
            if block_type == "tool_use" and getattr(block, "name", None) == expected_tool:
                tool_name = block.name
                candidate = getattr(block, "input", None)
                tool_input = candidate if isinstance(candidate, dict) else None
            elif block_type == "text":
                text = getattr(block, "text", None)
                if text:
                    texts.append(text)

        usage = getattr(raw, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        stop_reason = getattr(raw, "stop_reason", None) or "unknown"
        model_id = getattr(raw, "model", None) or self._config.model

        return LLMToolResponse(
            tool_name=tool_name,
            tool_input=tool_input,
            stop_reason=stop_reason,
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            text="\n".join(texts) if texts else None,
        )
