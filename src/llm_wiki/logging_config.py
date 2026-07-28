"""Centralised structlog configuration.

Call ``configure_logging()`` once at process startup (from both the FastAPI
app lifespan and the Celery worker bootstrap).  All output is JSON, one
record per line, with timestamps in ISO-8601 UTC.

Designed to be forward-compatible with OpenTelemetry: the JSON records
include the keys an OTel collector would expect (``timestamp``, ``level``,
``logger``, ``event``).  When OTel is added later, the same logger calls
will continue to work — only the renderer changes.
"""

from __future__ import annotations

import logging
import sys

import structlog

from llm_wiki.config import settings


def configure_logging() -> None:
    """Configure structlog + stdlib logging for JSON output to stderr.

    Idempotent: safe to call multiple times (``structlog.configure`` is
    itself idempotent; ``logging.basicConfig`` is guarded by ``force=True``
    which replaces any existing handlers on the root logger).

    Sets the stdlib root logger level from ``settings.log_level`` and routes
    everything through the structlog processor chain so that even third-party
    libraries (uvicorn, sqlalchemy, openai) produce structured JSON output.
    """
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Stdlib root logger → stderr with a bare format string.
    # structlog's JSONRenderer does the real work; the stdlib format is just
    # the transport layer.
    #
    # stderr (not stdout) on purpose: on this cluster filebeat ships only the
    # containers' stderr to Kibana — the worker's stdout, where these app logs
    # and tracebacks (pipeline_failed / pipeline_retry) would otherwise go, is
    # dropped, so failures were invisible. Routing app logs to stderr puts them
    # in the stream that is actually indexed. stderr is also line-buffered, so
    # records survive a hard worker kill (OOM/SIGKILL) instead of dying in an
    # unflushed buffer.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=level,
        force=True,
    )

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    # Quiet noisy third-party libraries to at least WARNING so they don't
    # drown out application logs.
    for noisy in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))
