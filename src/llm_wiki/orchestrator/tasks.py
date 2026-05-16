"""Celery task definitions — thin wrappers around the pipeline."""

import asyncio
from typing import Any

from celery import Celery
from celery.schedules import crontab

from llm_wiki.config import settings

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


@celery_app.task(bind=True, max_retries=3, name="llm_wiki.orchestrator.tasks.process_file")
def process_file_task(self: Any, file_id: str) -> None:
    """Celery task: run the ingestion pipeline for a single file.

    Retries up to 3 times with exponential backoff (4^attempt seconds).
    Implemented in LW-9.
    """
    from llm_wiki.orchestrator.pipeline import process_file  # local import avoids circular

    try:
        asyncio.run(process_file(file_id))
    except Exception as exc:
        # self is the bound Celery Task instance; retry raises Retry exception
        retry_count: int = self.request.retries
        raise self.retry(exc=exc, countdown=4**retry_count)


@celery_app.task(name="llm_wiki.orchestrator.tasks.run_lint")
def run_lint() -> None:
    """Celery task: trigger the weekly Lint Agent run.

    Implemented in LW-14/LW-15.
    """
    raise NotImplementedError("Implemented in LW-14 / LW-15")
