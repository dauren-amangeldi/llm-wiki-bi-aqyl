"""Request/task correlation without retaining prompts, credentials or bodies."""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

CORRELATION_KEYS = ("request_id", "run_id", "scenario_id", "operation_id")
ENTITY_KEYS = (
    "file_id",
    "case_id",
    "document_id",
    "artifact_id",
    "generation_id",
    "session_id",
    "consultation_id",
)
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")


def safe_id(value: Any) -> str | None:
    return value if isinstance(value, str) and _ID.fullmatch(value) else None


@dataclass
class Trace:
    fields: dict[str, Any] = field(default_factory=dict)
    outcome: str | None = None
    stream_done: bool = False
    stream_events: int = 0
    first_event_at: float | None = None


_trace: ContextVar[Trace | None] = ContextVar("observability_trace", default=None)


def context() -> dict[str, Any]:
    trace = _trace.get()
    return dict(trace.fields) if trace else {}


@contextmanager
def trace_scope(fields: dict[str, Any]) -> Iterator[Trace]:
    trace = Trace(fields=dict(fields))
    token = _trace.set(trace)
    try:
        yield trace
    finally:
        _trace.reset(token)


def bind_entities(**fields: Any) -> None:
    trace = _trace.get()
    if trace:
        trace.fields.update(
            {key: value for key, value in fields.items() if key in ENTITY_KEYS and safe_id(value)}
        )


def mark_outcome(outcome: str) -> None:
    trace = _trace.get()
    if trace and trace.outcome != "failed":
        trace.outcome = outcome


def observe_sse(payload: dict[str, Any]) -> None:
    """Observe metadata before serialization; never copy stream content to logs."""
    trace = _trace.get()
    if trace is None:
        return
    trace.stream_events += 1
    if trace.first_event_at is None:
        trace.first_event_at = time.perf_counter()
    if (
        payload.get("error")
        or payload.get("timeout")
        or payload.get("status") in ("FAILED", "ROLLED_BACK")
    ):
        mark_outcome("failed")
    content = payload.get("content")
    if isinstance(content, dict) and content.get("failed"):
        mark_outcome("degraded")
    if payload.get("done"):
        trace.stream_done = True
    if payload.get("refused") or payload.get("refusal"):
        mark_outcome("rejected")
    bind_entities(**{key: payload[key] for key in ENTITY_KEYS if key in payload})


def merge_trace(_logger: Any, _method: str, event: dict[str, Any]) -> dict[str, Any]:
    for key, value in context().items():
        event.setdefault(key, value)
    return event


def error_fields(exc: BaseException) -> dict[str, Any]:
    """Do not serialize provider error bodies: they may echo an API key."""
    result: dict[str, Any] = {"error_type": type(exc).__name__}
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        result["provider_status_code"] = status
    code = safe_id(getattr(exc, "code", None))
    if code:
        result["error_code"] = code
    return result


_SECRET = re.compile(r"(?i)\bsk-[a-z0-9_*\-]{4,}|\bBearer\s+[a-z0-9_.~+/=\-]+")
_SENSITIVE = {
    "authorization",
    "cookie",
    "set-cookie",
    "api_key",
    "access_token",
    "refresh_token",
    "password",
    "client_secret",
    "prompt",
    "messages",
}


def redact(_logger: Any, _method: str, event: dict[str, Any]) -> dict[str, Any]:
    def clean(value: Any) -> Any:
        if isinstance(value, str):
            return _SECRET.sub("[REDACTED]", value)
        if isinstance(value, dict):
            return {
                k: "[REDACTED]" if str(k).lower() in _SENSITIVE else clean(v)
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        return value

    return clean(event)


def new_id() -> str:
    return uuid4().hex[:16]
