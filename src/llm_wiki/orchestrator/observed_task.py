"""Correlate every publish and execution, including task-to-task dispatch."""

import inspect
import time
from typing import Any
from uuid import uuid4

import structlog
from celery import Task
from celery.exceptions import Ignore, Retry

from llm_wiki.observability import (
    CORRELATION_KEYS,
    ENTITY_KEYS,
    context,
    error_fields,
    safe_id,
    trace_scope,
)

logger = structlog.get_logger(__name__)


class ObservedTask(Task):
    abstract = True

    def _entities(self, args: Any, kwargs: Any) -> dict[str, str]:
        try:
            values = (
                inspect.signature(self.run).bind_partial(*(args or ()), **(kwargs or {})).arguments
            )
            return {
                key: value for key, value in values.items() if key in ENTITY_KEYS and safe_id(value)
            }
        except (TypeError, ValueError):
            return {}

    def apply_async(self, args=None, kwargs=None, task_id=None, **options):
        task_id = task_id or str(uuid4())
        fields = {key: value for key, value in context().items() if key in CORRELATION_KEYS}
        # Retry signatures carry original headers even when the worker context
        # has already unwound. Preserve them, but refresh the publish timestamp.
        headers = dict(options.pop("headers", None) or {})
        fields = {**headers.get("observability", {}), **fields, **self._entities(args, kwargs)}
        fields.setdefault("operation_id", task_id)
        fields["published_at"] = time.time()
        headers["observability"] = fields
        route = self.app.amqp.router.route(options, self.name, args, kwargs)
        queue = route.get("queue")
        queue_name = getattr(queue, "name", queue) or self.app.conf.task_default_queue
        metadata = {**fields, "task_id": task_id, "task_name": self.name, "queue": queue_name}
        # Celery's own task logs must not serialize task arguments/results.
        options.setdefault("argsrepr", "<redacted>")
        options.setdefault("kwargsrepr", "<redacted>")
        try:
            result = super().apply_async(
                args=args, kwargs=kwargs, task_id=task_id, headers=headers, **options
            )
        except Exception as exc:
            logger.error("task_enqueue_failed", **metadata, **error_fields(exc))
            raise
        logger.info("task_queued", **metadata)
        return result

    def __call__(self, *args, **kwargs):
        from llm_wiki.logging_config import configure_logging

        configure_logging()
        request = self.request
        headers = (request.headers or {}).get("observability", {})
        fields = {
            key: value
            for key, value in headers.items()
            if key in (*CORRELATION_KEYS, *ENTITY_KEYS) and safe_id(value)
        }
        fields.update(self._entities(args, kwargs))
        fields.update(
            task_id=request.id or str(uuid4()),
            task_name=self.name,
            queue=(request.delivery_info or {}).get("routing_key"),
            retry_count=request.retries or 0,
        )
        fields.setdefault("operation_id", fields["task_id"])
        published = headers.get("published_at")
        wait_ms = (
            max(0, round((time.time() - published) * 1000, 2))
            if isinstance(published, (int, float))
            else None
        )
        started = time.perf_counter()
        previous = structlog.contextvars.get_contextvars()
        structlog.contextvars.clear_contextvars()
        with trace_scope(fields) as trace:
            logger.info("task_execution_started", queue_wait_ms=wait_ms, **fields)
            outcome = "success"
            failure = {}
            try:
                # Celery's worker already pushed its request. Task.__call__
                # would push an empty one and hide its id/retry/delivery info.
                result = self.run(*args, **kwargs)
                if isinstance(result, dict):
                    status = result.get("status")
                    if status in ("failed", "error", "timeout"):
                        outcome = "failed"
                    elif status in ("skipped", "not_found", "already_tagged", "stale"):
                        outcome = "skipped"
                    elif status == "degraded":
                        outcome = "degraded"
                return result
            except Retry:
                outcome = "retry"
                raise
            except Ignore:
                outcome = "skipped"
                raise
            except BaseException as exc:
                outcome = "failed"
                failure = error_fields(exc)
                raise
            finally:
                logger.info(
                    "task_execution_finished",
                    outcome=outcome if outcome in ("failed", "retry") else trace.outcome or outcome,
                    duration_ms=round((time.perf_counter() - started) * 1000, 2),
                    queue_wait_ms=wait_ms,
                    **fields,
                    **failure,
                )
                structlog.contextvars.clear_contextvars()
                structlog.contextvars.bind_contextvars(**previous)
