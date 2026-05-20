"""Celery task definitions — thin wrappers around the pipeline."""

import asyncio
from typing import Any

import openai
import structlog
from celery import Celery
from celery.schedules import crontab

from llm_wiki.config import settings

logger = structlog.get_logger(__name__)

celery_app = Celery(
    "llm_wiki",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    worker_concurrency=1,  # MVP: single worker to avoid index.md race conditions
    beat_schedule={
        "weekly-lint": {
            "task": "llm_wiki.orchestrator.tasks.run_lint",
            "schedule": crontab(minute=0, hour=3, day_of_week=1),  # Mon 03:00 UTC
        },
    },
)

# Errors that must NEVER be retried at the Celery level — retrying would be
# pointless (the model won't appear by itself, credentials won't fix themselves).
_PERMANENT_ERRORS: tuple[type[Exception], ...] = (
    openai.NotFoundError,       # model not pulled in Ollama / wrong model name
    openai.AuthenticationError, # bad API key
    openai.PermissionDeniedError,
)


@celery_app.task(bind=True, max_retries=3, name="llm_wiki.orchestrator.tasks.process_file")
def process_file_task(self: Any, file_id: str) -> None:
    """Celery task: run the ingestion pipeline for a single file.

    Uses ``asyncio.Runner`` (Python 3.11+) so the same event loop is reused
    across the coroutine's full lifetime, which prevents the
    ``RuntimeError: Event loop is closed`` that occurs when an httpx
    ``AsyncClient`` is torn down after a failed request on a *different* loop
    (the old ``asyncio.run()`` creates a new loop per call and closes it
    immediately, before httpx can clean up).

    Permanent errors (model not found, bad auth) are **not** retried — they
    are logged immediately and re-raised so the task fails fast with a clear
    message.

    Args:
        file_id: UUID of the file to process.
    """
    from llm_wiki.orchestrator.pipeline import process_file  # local import avoids circular

    try:
        # asyncio.Runner keeps the loop alive across await points AND across
        # the context-manager lifetime, so httpx can close connections cleanly.
        with asyncio.Runner() as runner:
            runner.run(process_file(file_id))
    except _PERMANENT_ERRORS as exc:
        # No point retrying — log and fail permanently.
        logger.error(
            "pipeline_failed_permanent",
            file_id=file_id,
            error=str(exc),
            hint=(
                "Model not found — run `docker compose exec ollama ollama pull <model>`"
                if isinstance(exc, openai.NotFoundError)
                else "Check your API credentials."
            ),
        )
        raise  # let Celery mark the task as FAILURE without scheduling retries
    except Exception as exc:
        retry_count: int = self.request.retries
        logger.warning(
            "pipeline_retry",
            file_id=file_id,
            attempt=retry_count + 1,
            error=str(exc),
        )
        raise self.retry(exc=exc, countdown=4**retry_count)


@celery_app.task(name="llm_wiki.orchestrator.tasks.run_lint")
def run_lint() -> None:
    """Celery task: trigger the weekly Lint Agent run.

    Implemented in LW-14/LW-15.
    """
    raise NotImplementedError("Implemented in LW-14 / LW-15")
