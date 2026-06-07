"""Unit tests for logging_config (LW-17 lite)."""

from __future__ import annotations

import io
import json
import logging
from contextlib import redirect_stdout

import structlog

from llm_wiki.logging_config import configure_logging


def test_log_output_is_valid_json() -> None:
    configure_logging()
    buf = io.StringIO()
    with redirect_stdout(buf):
        structlog.get_logger("test").info("smoke_test", foo="bar", n=1)
    line = buf.getvalue().strip()
    assert line, "no log line produced"
    record = json.loads(line)
    assert record["event"] == "smoke_test"
    assert record["foo"] == "bar"
    assert record["n"] == 1
    assert "timestamp" in record
    assert record["level"] == "info"


def test_contextvars_appear_in_output() -> None:
    configure_logging()
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id="abc123", file_id="f-1")
    buf = io.StringIO()
    with redirect_stdout(buf):
        structlog.get_logger("test").info("with_context")
    structlog.contextvars.clear_contextvars()
    record = json.loads(buf.getvalue().strip())
    assert record["request_id"] == "abc123"
    assert record["file_id"] == "f-1"


def test_log_level_respected(monkeypatch: object) -> None:
    import llm_wiki.config as cfg_mod

    monkeypatch.setattr(cfg_mod.settings, "log_level", "WARNING")  # type: ignore[attr-defined]
    configure_logging()
    buf = io.StringIO()
    with redirect_stdout(buf):
        structlog.get_logger("test").info("should_be_suppressed")
        structlog.get_logger("test").warning("should_appear")
    lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0])["event"] == "should_appear"


def test_configure_logging_is_idempotent() -> None:
    """Calling configure_logging() twice must not raise or duplicate handlers."""
    configure_logging()
    before = len(logging.getLogger().handlers)
    configure_logging()
    after = len(logging.getLogger().handlers)
    # basicConfig with force=True replaces handlers — count stays stable.
    assert after == before


def test_noisy_libs_are_quieted() -> None:
    configure_logging()
    for lib in ("httpx", "httpcore", "openai", "urllib3"):
        lib_level = logging.getLogger(lib).level
        assert lib_level >= logging.WARNING, (
            f"Expected {lib} to be at least WARNING, got {logging.getLevelName(lib_level)}"
        )
