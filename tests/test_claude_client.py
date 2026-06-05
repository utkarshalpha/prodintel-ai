"""Unit tests for ClaudeToolClient and its configuration."""

from __future__ import annotations

import pytest

from app.ai_clients.claude_client import ClaudeToolClient
from app.ai_clients.config import DEFAULT_MODEL, ClaudeClientConfig
from app.ai_clients.errors import ClaudeConfigurationError, ClaudeFatalError
from app.ai_runtime.errors import LLMClientError
from app.ai_runtime.interfaces import LLMMessage, ToolSpec

from tests.anthropic_fakes import (
    FakeAnthropic,
    auth_error,
    bad_request_error,
    fake_message,
    rate_limit_error,
    server_error,
    text_block,
    timeout_error,
    tool_message,
)

TOOL = ToolSpec(name="emit_x", description="emit x", input_schema={"type": "object", "properties": {}})


def _config() -> ClaudeClientConfig:
    return ClaudeClientConfig(api_key="sk-test", model="claude-test", max_tokens=512, timeout_seconds=5)


def _client(behaviors) -> tuple[ClaudeToolClient, FakeAnthropic]:
    fake = FakeAnthropic(behaviors)
    return ClaudeToolClient(_config(), client=fake), fake


def _msgs(*pairs: tuple[str, str]) -> list[LLMMessage]:
    return [LLMMessage(role=r, content=c) for r, c in pairs]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def test_from_env_requires_api_key() -> None:
    with pytest.raises(ClaudeConfigurationError):
        ClaudeClientConfig.from_env({})


def test_from_env_reads_values() -> None:
    cfg = ClaudeClientConfig.from_env(
        {"ANTHROPIC_API_KEY": "sk-abc", "ANTHROPIC_MODEL": "claude-z", "ANTHROPIC_MAX_TOKENS": "999"}
    )
    assert cfg.api_key.get_secret_value() == "sk-abc"
    assert cfg.model == "claude-z"
    assert cfg.max_tokens == 999
    assert cfg.max_retries == 0  # harness owns retries


def test_api_key_is_not_leaked_in_repr() -> None:
    cfg = _config()
    assert "sk-test" not in repr(cfg)


def test_default_model_used_when_unset() -> None:
    cfg = ClaudeClientConfig.from_env({"ANTHROPIC_API_KEY": "k"})
    assert cfg.model == DEFAULT_MODEL


# --------------------------------------------------------------------------- #
# Successful tool response mapping
# --------------------------------------------------------------------------- #
def test_successful_tool_response_is_mapped() -> None:
    client, _ = _client([tool_message("emit_x", {"a": 1}, text="thinking", input_tokens=111, output_tokens=22)])
    resp = client.complete(system="sys", messages=_msgs(("user", "hi")), tool=TOOL)

    assert resp.tool_name == "emit_x"
    assert resp.tool_input == {"a": 1}
    assert resp.stop_reason == "tool_use"
    assert resp.model_id == "claude-test"
    assert resp.input_tokens == 111
    assert resp.output_tokens == 22
    assert resp.text == "thinking"


def test_forced_tool_choice_and_payload_are_sent() -> None:
    client, fake = _client([tool_message("emit_x", {"a": 1})])
    client.complete(system="SYSTEM", messages=_msgs(("user", "hi")), tool=TOOL)

    sent = fake.messages.calls[0]
    assert sent["tool_choice"] == {"type": "tool", "name": "emit_x"}
    assert sent["tools"][0]["input_schema"] == TOOL.input_schema
    assert sent["model"] == "claude-test"
    assert sent["max_tokens"] == 512
    assert sent["system"] == "SYSTEM"


def test_consecutive_same_role_messages_are_coalesced() -> None:
    client, fake = _client([tool_message("emit_x", {"a": 1})])
    client.complete(system="s", messages=_msgs(("user", "first"), ("user", "second")), tool=TOOL)

    assert fake.messages.calls[0]["messages"] == [{"role": "user", "content": "first\n\nsecond"}]


def test_no_tool_use_block_yields_none_input() -> None:
    client, _ = _client([fake_message(content=[text_block("no tool here")], stop_reason="end_turn")])
    resp = client.complete(system="s", messages=_msgs(("user", "hi")), tool=TOOL)

    assert resp.tool_name is None
    assert resp.tool_input is None
    assert resp.stop_reason == "end_turn"
    assert resp.text == "no tool here"


# --------------------------------------------------------------------------- #
# Error classification: retryable vs fatal
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("error_factory", [timeout_error, rate_limit_error, server_error])
def test_transient_errors_raise_llm_client_error(error_factory) -> None:
    client, _ = _client([error_factory()])
    with pytest.raises(LLMClientError):
        client.complete(system="s", messages=_msgs(("user", "hi")), tool=TOOL)


@pytest.mark.parametrize("error_factory", [auth_error, bad_request_error])
def test_permanent_errors_raise_fatal(error_factory) -> None:
    client, _ = _client([error_factory()])
    with pytest.raises(ClaudeFatalError):
        client.complete(system="s", messages=_msgs(("user", "hi")), tool=TOOL)
