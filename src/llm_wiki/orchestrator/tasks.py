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
        # LW-15: weekly semantic audit via OpenAI Batch API (-50% cost, 24 h SLA)
        "weekly-audit": {
            "task": "llm_wiki.orchestrator.tasks.run_weekly_audit",
            "schedule": crontab(minute=0, hour=3, day_of_week=1),  # Mon 03:00 UTC
        },
    },
)

# Errors that must NEVER be retried at the Celery level — retrying would be
# pointless (the model won't appear by itself, credentials won't fix themselves).
_PERMANENT_ERRORS: tuple[type[Exception], ...] = (
    openai.NotFoundError,       # wrong model name / model unavailable
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
    import logging as _stdlib_logging

    # Stdlib log before any local imports — distinguishes Celery "received"
    # from actually entering the task body (branch A vs B in runbook).
    _stdlib_logging.getLogger(__name__).info(
        "celery_task_entered file_id=%s celery_task_id=%s",
        file_id,
        getattr(self.request, "id", "?"),
    )

    from llm_wiki.logging_config import configure_logging
    from llm_wiki.orchestrator.pipeline import process_file  # local import avoids circular

    # Ensure structured JSON logging is active in the worker process.
    configure_logging()

    # Bind file_id to the contextvar log context so every nested call —
    # pipeline stages, LLM client, storage helpers — automatically includes
    # it without manual .bind(file_id=...) calls.
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        file_id=file_id,
        task_name="process_file",
    )
    log = structlog.get_logger(__name__)
    log.info("task_started")

    try:
        # asyncio.Runner keeps the loop alive across await points AND across
        # the context-manager lifetime, so httpx can close connections cleanly.
        with asyncio.Runner() as runner:
            # Suppress harmless "Event loop is closed" from httpx's TLS
            # cleanup tasks that may outlive the Runner's loop.
            _original_handler = runner.get_loop().get_exception_handler()

            def _suppress_loop_closed(
                loop: asyncio.AbstractEventLoop,
                context: dict,  # type: ignore[type-arg]
            ) -> None:
                exc = context.get("exception")
                if isinstance(exc, RuntimeError) and "Event loop is closed" in str(exc):
                    return
                if _original_handler is not None:
                    _original_handler(loop, context)
                else:
                    loop.default_exception_handler(context)

            runner.get_loop().set_exception_handler(_suppress_loop_closed)
            runner.run(process_file(file_id))

        log.info("task_completed")
    except _PERMANENT_ERRORS as exc:
        # No point retrying — log and fail permanently.
        log.error(
            "pipeline_failed_permanent",
            error=str(exc),
            exc_info=True,
            hint=(
                "Model not found — check OPENAI_MODEL / ANTHROPIC_MODEL is valid"
                if isinstance(exc, openai.NotFoundError)
                else "Check your API credentials."
            ),
        )
        raise  # let Celery mark the task as FAILURE without scheduling retries
    except Exception as exc:
        retry_count: int = self.request.retries
        log.warning(
            "pipeline_retry",
            attempt=retry_count + 1,
            error=str(exc),
            exc_info=True,
        )
        raise self.retry(exc=exc, countdown=4**retry_count)
    finally:
        structlog.contextvars.clear_contextvars()


@celery_app.task(name="llm_wiki.orchestrator.tasks.run_weekly_audit")
def run_weekly_audit(
    mode: str = "batch",
    dry_run: bool = False,
    sample: int | None = None,
    slugs: list[str] | None = None,
) -> dict[str, object]:
    """Celery task: run the LLM Auditor over the entire wiki.

    Triggered weekly by Celery Beat (Monday 03:00 UTC) using Batch API for
    -50% cost.  Also callable manually from the API or CLI (with ``mode="sync"``
    for immediate results).

    Args:
        mode: ``"batch"`` (Batch API, 24 h SLA) or ``"sync"`` (completions).
        dry_run: If True, run the auditor but do not write to ``issues.md``.
        sample: If set, only audit this many randomly-selected pages.
        slugs: If set, only audit these specific page slugs.

    Returns:
        ``{"issues_found": int, "mode": str, "dry_run": bool}``
    """
    from llm_wiki.logging_config import configure_logging

    configure_logging()
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(task_name="run_weekly_audit", mode=mode)
    _audit_log = structlog.get_logger(__name__)
    _audit_log.info("task_started")

    import asyncio
    import random
    from pathlib import Path
    from typing import Literal

    from llm_wiki.agents.auditor import AuditorAgent
    from llm_wiki.config import settings
    from llm_wiki.llm.client import LLMClient
    from llm_wiki.quality.issues_writer import upsert_section
    from llm_wiki.quality.models import IssueSection

    wiki_dir: Path = settings.wiki_dir
    wiki_pages: list[tuple[str, str]] = []
    if wiki_dir.exists():
        for md_file in sorted(wiki_dir.glob("*.md")):
            wiki_pages.append((md_file.stem, md_file.read_text(encoding="utf-8")))

    # Optional filtering
    if slugs:
        wiki_pages = [(s, c) for s, c in wiki_pages if s in set(slugs)]
    if sample and len(wiki_pages) > sample:
        wiki_pages = random.sample(wiki_pages, sample)

    if not wiki_pages:
        logger.info("weekly_audit_no_pages")
        return {"issues_found": 0, "mode": mode, "dry_run": dry_run}

    # Build related pairs from ChromaDB (cosine > 0.6)
    related_pairs: list[tuple[str, str]] = []
    try:
        from llm_wiki.llm.embeddings import EmbeddingStore

        llm_tmp = LLMClient()
        emb_store = EmbeddingStore(
            chroma_path=settings.chroma_dir, llm_client=llm_tmp
        )
        for slug, _ in wiki_pages:
            hits = emb_store.query(
                slug,
                top_k=5,
                file_id="weekly-audit",
            )
            for hit in hits:
                if hit.score >= 0.6 and hit.slug != slug:
                    pair = tuple(sorted([slug, hit.slug]))
                    if pair not in related_pairs:
                        related_pairs.append(pair)  # type: ignore[arg-type]
    except Exception as emb_exc:  # noqa: BLE001
        logger.warning("weekly_audit_embedding_pairs_failed", error=str(emb_exc))

    llm = LLMClient()

    def _run() -> list[object]:
        async def _inner() -> list[object]:
            agent = AuditorAgent(llm)
            return await agent.run(
                wiki_pages=wiki_pages,
                related_pairs=related_pairs,  # type: ignore[arg-type]
                mode=mode,  # type: ignore[arg-type]
            )

        with asyncio.Runner() as runner:
            return runner.run(_inner())

    try:
        issues = _run()
    except Exception as exc:
        logger.error("weekly_audit_failed", error=str(exc))
        raise
    finally:
        # Close LLM client synchronously — we are not in async context here
        try:
            asyncio.run(llm.aclose())
        except RuntimeError:
            pass

    if not dry_run:
        upsert_section(
            issues_path=settings.issues_path,
            section=IssueSection.LLM_FLAGGED,
            issues=list(issues),  # type: ignore[arg-type]
        )

    counts: dict[str, int] = {}
    for issue in issues:
        k = str(getattr(issue, "kind", "unknown"))
        counts[k] = counts.get(k, 0) + 1

    _audit_log.info(
        "weekly_audit_done",
        issues_found=len(issues),
        dry_run=dry_run,
        **counts,
    )
    structlog.contextvars.clear_contextvars()
    return {"issues_found": len(issues), "mode": mode, "dry_run": dry_run}
