"""Tests for structured logging, request context, and correlation ids."""

from __future__ import annotations

import io
import json
import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.ai_contracts.enums import StakeholderType
from app.api.middleware import CorrelationIdMiddleware
from app.observability.logging import (
    ContextFilter,
    JsonFormatter,
    configure_logging,
    get_context,
    log_context,
    new_correlation_id,
)
from app.repositories.parsed_signal_repository import ParsedSignalRepository
from app.repositories.signal_repository import SignalRepository
from app.services.signal_service import SignalService
from app.stages.stage1.runner import build_stage1_runner

from tests.app_helpers import Stage1FakeClient, make_engine, make_session_factory


# --------------------------------------------------------------------------- #
# JsonFormatter
# --------------------------------------------------------------------------- #
def test_json_formatter_emits_required_fields_and_extra() -> None:
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="app.test", level=logging.INFO, pathname=__file__, lineno=1,
        msg="hello", args=(), exc_info=None,
    )
    record.event = "unit_test"
    record.signal_id = "abc-123"

    payload = json.loads(formatter.format(record))
    assert payload["level"] == "INFO"
    assert payload["logger"] == "app.test"
    assert payload["message"] == "hello"
    assert payload["event"] == "unit_test"
    assert payload["signal_id"] == "abc-123"
    assert "timestamp" in payload


def test_json_formatter_includes_exception() -> None:
    formatter = JsonFormatter()
    try:
        raise ValueError("boom")
    except ValueError:
        record = logging.LogRecord(
            name="app.test", level=logging.ERROR, pathname=__file__, lineno=1,
            msg="failed", args=(), exc_info=True,
        )
        import sys

        record.exc_info = sys.exc_info()
    payload = json.loads(formatter.format(record))
    assert "ValueError: boom" in payload["exception"]


# --------------------------------------------------------------------------- #
# Context binding
# --------------------------------------------------------------------------- #
def test_log_context_binds_and_resets() -> None:
    assert get_context() == {}
    with log_context(correlation_id="cid-1"):
        assert get_context()["correlation_id"] == "cid-1"
    assert get_context() == {}


def test_context_filter_injects_context_onto_record() -> None:
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "m", (), None)
    with log_context(correlation_id="cid-9"):
        assert ContextFilter().filter(record) is True
    assert record.correlation_id == "cid-9"


def test_new_correlation_id_is_unique() -> None:
    assert new_correlation_id() != new_correlation_id()


def test_configure_logging_is_idempotent() -> None:
    root = logging.getLogger()
    before = list(root.handlers)
    try:
        configure_logging(stream=io.StringIO())
        configure_logging(stream=io.StringIO())
        prodintel = [h for h in root.handlers if getattr(h, "_prodintel", False)]
        assert len(prodintel) == 1  # idempotent: not stacked
    finally:
        root.handlers = before


def test_configure_logging_emits_json() -> None:
    root = logging.getLogger()
    before = list(root.handlers)
    stream = io.StringIO()
    try:
        configure_logging(stream=stream)
        logging.getLogger("app.test").info("structured", extra={"event": "x"})
        for handler in root.handlers:
            handler.flush()
        line = stream.getvalue().strip().splitlines()[-1]
        assert json.loads(line)["event"] == "x"
    finally:
        root.handlers = before


# --------------------------------------------------------------------------- #
# Correlation-id middleware
# --------------------------------------------------------------------------- #
@pytest.fixture
def ping_client() -> TestClient:
    app = FastAPI()
    app.add_middleware(CorrelationIdMiddleware)

    @app.get("/ping")
    def ping() -> dict[str, bool]:
        return {"ok": True}

    return TestClient(app)


def test_middleware_generates_correlation_id(ping_client: TestClient) -> None:
    resp = ping_client.get("/ping")
    assert resp.status_code == 200
    assert resp.headers.get("x-request-id")


def test_middleware_echoes_inbound_correlation_id(ping_client: TestClient) -> None:
    resp = ping_client.get("/ping", headers={"X-Request-ID": "trace-42"})
    assert resp.headers["x-request-id"] == "trace-42"


def test_middleware_logs_request(ping_client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="app.api.request"):
        ping_client.get("/ping")
    events = [r for r in caplog.records if getattr(r, "event", None) == "http_request"]
    assert events and events[0].status_code == 200


# --------------------------------------------------------------------------- #
# Service-level structured logging (integration)
# --------------------------------------------------------------------------- #
def test_analyze_emits_stage_and_db_write_logs(caplog: pytest.LogCaptureFixture) -> None:
    session = make_session_factory(make_engine())()
    service = SignalService(
        session,
        SignalRepository(session),
        ParsedSignalRepository(session),
        build_stage1_runner(Stage1FakeClient(grounded=True)),
    )
    signal = service.create_signal(source_type=StakeholderType.CUSTOMER, raw_text="Checkout fails on pay").signal

    with caplog.at_level(logging.INFO, logger="app.services.signal_service"):
        service.analyze_signal(signal.id)

    events = {getattr(r, "event", None) for r in caplog.records}
    assert "stage_started" in events
    assert "stage_succeeded" in events
    assert "db_write" in events
    session.close()
