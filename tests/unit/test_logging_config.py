"""JSON logging, correlation and secret redaction on stderr."""
import json
import logging

import structlog

from llm_wiki.logging_config import configure_logging
from llm_wiki.observability import trace_scope


def test_log_output_is_valid_json(capsys):
    configure_logging()
    structlog.get_logger("test").info("smoke_test", foo="bar", n=1)
    record = json.loads(capsys.readouterr().err)
    assert record["event"] == "smoke_test"
    assert record["foo"] == "bar"
    assert record["n"] == 1
    assert record["timestamp"]
    assert record["level"] == "info"


def test_contextvars_appear_in_output(capsys):
    configure_logging()
    with trace_scope({"request_id": "abc123", "run_id": "load-1"}):
        structlog.get_logger("test").info("with_context")
    record = json.loads(capsys.readouterr().err)
    assert record["request_id"] == "abc123"
    assert record["run_id"] == "load-1"


def test_log_level_respected(monkeypatch, capsys):
    from llm_wiki.config import settings
    monkeypatch.setattr(settings, "log_level", "WARNING")
    configure_logging()
    structlog.get_logger("test").info("should_be_suppressed")
    structlog.get_logger("test").warning("should_appear")
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["event"] == "should_appear"


def test_stdlib_is_json_and_redacts_credentials(capsys):
    configure_logging()
    with trace_scope({"request_id": "same-id"}):
        logging.getLogger("worker").warning("bad key sk-testsecret and Bearer token.secret")
        structlog.get_logger("test").warning("failed", api_key="private-value", prompt="private prompt")
    raw = capsys.readouterr().err
    records = [json.loads(line) for line in raw.splitlines()]
    assert all(r["request_id"] == "same-id" for r in records)
    assert "sk-testsecret" not in raw and "token.secret" not in raw
    assert "private-value" not in raw and "private prompt" not in raw


def test_configure_logging_is_idempotent():
    configure_logging()
    count = len(logging.getLogger().handlers)
    configure_logging()
    assert len(logging.getLogger().handlers) == count


def test_noisy_libs_are_quieted():
    configure_logging()
    for name in ("httpx", "httpcore", "openai", "urllib3"):
        assert logging.getLogger(name).level >= logging.WARNING
