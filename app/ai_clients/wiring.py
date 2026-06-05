"""Startup wiring and dependency-injection integration for the Claude client.

Connects the adapter to the existing application without modifying any existing
file. The FastAPI dependency ``get_stage1_runner`` already reads
``app.state.stage1_runner``; these helpers construct a Claude-backed runner and set
that attribute.

A production retry policy with real backoff is used by default (the harness's
deterministic policy plus ``time.sleep``), so transient API errors are retried with
exponential backoff. The deterministic, zero-delay default remains available for
tests.
"""

from __future__ import annotations

import time

from fastapi import FastAPI

from app.ai_clients.claude_client import ClaudeToolClient
from app.ai_clients.config import ClaudeClientConfig
from app.ai_runtime.retry_policy import RetryPolicy
from app.api.app import create_app
from app.observability.logging import configure_logging
from app.stages.stage1.runner import Stage1SignalRunner, build_stage1_runner
from app.stages.stage2.runner import Stage2FeatureRunner, build_stage2_runner
from app.stages.stage3.runner import Stage3ConflictRunner, build_stage3_runner

__all__ = [
    "make_claude_stage1_runner",
    "make_claude_stage2_runner",
    "make_claude_stage3_runner",
    "attach_claude_runner",
    "create_production_app",
]

#: Default production retry policy: 3 attempts with capped exponential backoff.
_PRODUCTION_RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    base_delay_seconds=0.5,
    backoff_multiplier=2.0,
    max_delay_seconds=8.0,
)


def make_claude_stage1_runner(
    config: ClaudeClientConfig | None = None,
    *,
    retry_policy: RetryPolicy | None = None,
) -> Stage1SignalRunner:
    """Build a Stage 1 runner backed by a real Claude client.

    Reads config from the environment if none is supplied. Wires ``time.sleep`` so
    the retry policy's backoff actually waits in production.
    """

    resolved_config = config or ClaudeClientConfig.from_env()
    client = ClaudeToolClient(resolved_config)
    return build_stage1_runner(
        client,
        retry_policy=retry_policy or _PRODUCTION_RETRY_POLICY,
        sleep=time.sleep,
    )


def make_claude_stage2_runner(
    config: ClaudeClientConfig | None = None,
    *,
    retry_policy: RetryPolicy | None = None,
) -> Stage2FeatureRunner:
    """Build a Stage 2 runner backed by a real Claude client."""

    resolved_config = config or ClaudeClientConfig.from_env()
    client = ClaudeToolClient(resolved_config)
    return build_stage2_runner(
        client,
        retry_policy=retry_policy or _PRODUCTION_RETRY_POLICY,
        sleep=time.sleep,
    )


def make_claude_stage3_runner(
    config: ClaudeClientConfig | None = None,
    *,
    retry_policy: RetryPolicy | None = None,
) -> Stage3ConflictRunner:
    """Build a Stage 3 runner backed by a real Claude client."""

    resolved_config = config or ClaudeClientConfig.from_env()
    client = ClaudeToolClient(resolved_config)
    return build_stage3_runner(
        client,
        retry_policy=retry_policy or _PRODUCTION_RETRY_POLICY,
        sleep=time.sleep,
    )


def attach_claude_runner(
    app: FastAPI,
    *,
    config: ClaudeClientConfig | None = None,
    retry_policy: RetryPolicy | None = None,
) -> Stage1SignalRunner:
    """Construct the Claude-backed Stage 1/2/3 runners and register them on app.state.

    These are the integration points with the existing dependency injection:
    ``get_stage1_runner`` / ``get_stage2_runner`` / ``get_stage3_runner`` read
    ``app.state.stage1_runner`` / ``stage2_runner`` / ``stage3_runner`` respectively.
    """

    resolved = config or ClaudeClientConfig.from_env()
    stage1 = make_claude_stage1_runner(resolved, retry_policy=retry_policy)
    stage2 = make_claude_stage2_runner(resolved, retry_policy=retry_policy)
    stage3 = make_claude_stage3_runner(resolved, retry_policy=retry_policy)
    app.state.stage1_runner = stage1
    app.state.stage2_runner = stage2
    app.state.stage3_runner = stage3
    return stage1


def create_production_app(*, config: ClaudeClientConfig | None = None) -> FastAPI:
    """Compose the existing app with live Claude-backed Stage 1 and Stage 2 runners."""

    configure_logging()
    app = create_app()
    attach_claude_runner(app, config=config)
    return app
