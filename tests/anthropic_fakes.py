"""Fakes for the Anthropic SDK, so the Claude adapter can be tested with no network.

Provides a ``FakeAnthropic`` whose ``messages.create`` replays a script of return
values (fake messages) and raised exceptions, plus builders for fake tool-use
messages and real Anthropic error instances.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Sequence

import anthropic
import httpx


# --------------------------------------------------------------------------- #
# Fake SDK client
# --------------------------------------------------------------------------- #
class _FakeMessages:
    def __init__(self, behaviors: Sequence[Any]) -> None:
        self._behaviors = list(behaviors)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self._behaviors:
            raise AssertionError("FakeAnthropic ran out of scripted behaviors")
        behavior = self._behaviors.pop(0)
        if isinstance(behavior, Exception):
            raise behavior
        return behavior


class FakeAnthropic:
    """Stand-in for ``anthropic.Anthropic`` exposing ``.messages.create``."""

    def __init__(self, behaviors: Sequence[Any]) -> None:
        self.messages = _FakeMessages(behaviors)


# --------------------------------------------------------------------------- #
# Fake message/block builders
# --------------------------------------------------------------------------- #
def tool_use_block(name: str, tool_input: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", name=name, input=tool_input)


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def fake_message(
    *,
    content: list[SimpleNamespace],
    stop_reason: str = "tool_use",
    model: str = "claude-test",
    input_tokens: int = 100,
    output_tokens: int = 20,
) -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        model=model,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def tool_message(
    name: str,
    tool_input: dict[str, Any],
    *,
    text: str | None = None,
    stop_reason: str = "tool_use",
    model: str = "claude-test",
    input_tokens: int = 100,
    output_tokens: int = 20,
) -> SimpleNamespace:
    blocks: list[SimpleNamespace] = []
    if text:
        blocks.append(text_block(text))
    blocks.append(tool_use_block(name, tool_input))
    return fake_message(
        content=blocks,
        stop_reason=stop_reason,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


# --------------------------------------------------------------------------- #
# Real Anthropic exception builders (constructed with minimal httpx objects)
# --------------------------------------------------------------------------- #
def _request() -> httpx.Request:
    return httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def _response(status_code: int) -> httpx.Response:
    return httpx.Response(status_code, request=_request())


def timeout_error() -> anthropic.APITimeoutError:
    return anthropic.APITimeoutError(request=_request())


def rate_limit_error() -> anthropic.RateLimitError:
    return anthropic.RateLimitError("rate limited", response=_response(429), body=None)


def server_error() -> anthropic.InternalServerError:
    return anthropic.InternalServerError("server error", response=_response(503), body=None)


def auth_error() -> anthropic.AuthenticationError:
    return anthropic.AuthenticationError("bad key", response=_response(401), body=None)


def bad_request_error() -> anthropic.BadRequestError:
    return anthropic.BadRequestError("bad request", response=_response(400), body=None)
