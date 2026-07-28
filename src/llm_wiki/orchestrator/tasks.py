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
    # Keep Celery's hands off sys.stdout/sys.stderr. Its default
    # (worker_redirect_stdouts=True) swaps them for a LoggingProxy that
    # swallows structlog's PrintLogger writes — so pipeline_failed and the
    # pipeline_retry traceback never reached the container's stdout/stderr and
    # were invisible in BOTH Kibana and `kubectl logs`, even though the task's
    # except clause ran (the file went to status=FAILED). With this off,
    # structlog writes straight to the real stderr (see logging_config) and the
    # failure reason + traceback are shipped as-is.
    worker_redirect_stdouts=False,
    worker_concurrency=1,  # MVP: single worker to avoid index.md race conditions
    # At-least-once delivery: ack only AFTER the task finishes, and re-queue if
    # the worker dies mid-task (OOM / SIGKILL / eviction). Combined with the
    # persisted state_history the pipeline resumes instead of duplicating work.
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # With acks_late, keep at most one un-acked message per worker so a crash
    # re-queues a single task, not a whole prefetched batch.
    worker_prefetch_multiplier=1,
    # Redis re-delivers an un-acked message after this many seconds. Must exceed
    # the longest possible pipeline run so a slow (not dead) task is not
    # re-queued while still processing.
    broker_transport_options={"visibility_timeout": 3600},
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


@celery_app.task(
    bind=True,
    max_retries=3,
    acks_late=True,
    reject_on_worker_lost=True,
    name="llm_wiki.orchestrator.tasks.process_file",
)
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

    from llm_wiki.agents.auditor import AuditorAgent
    from llm_wiki.config import settings
    from llm_wiki.llm.client import LLMClient
    from llm_wiki.quality.issues_writer import upsert_section
    from llm_wiki.quality.models import IssueSection

    from llm_wiki.storage import wiki_store

    wiki_pages: list[tuple[str, str]] = wiki_store.get_all_pages()

    # Optional filtering
    if slugs:
        wiki_pages = [(s, c) for s, c in wiki_pages if s in set(slugs)]
    if sample and len(wiki_pages) > sample:
        wiki_pages = random.sample(wiki_pages, sample)

    if not wiki_pages:
        logger.info("weekly_audit_no_pages")
        return {"issues_found": 0, "mode": mode, "dry_run": dry_run}

    # Build related pairs from pgvector (cosine > 0.6)
    related_pairs: list[tuple[str, str]] = []
    try:
        from llm_wiki.llm.embeddings import EmbeddingStore

        llm_tmp = LLMClient()
        emb_store = EmbeddingStore(
            llm_client=llm_tmp
        )
        for slug, _ in wiki_pages:
            hits = emb_store.query(
                slug,
                top_k=5,
                file_id="weekly-audit",
            )
            for hit in hits:
                if hit.similarity >= 0.6 and hit.slug != slug:
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


@celery_app.task(name="llm_wiki.orchestrator.tasks.autotag_case")
def autotag_case(case_id: str, force: bool = False) -> dict[str, object]:
    """Classify a case against the fixed taxonomy and set its tags.

    Skips a case that already has tags (unless ``force``) so it never clobbers a
    user's manual edits. Best-effort — any failure leaves the tags untouched.
    """
    import asyncio

    from llm_wiki.logging_config import configure_logging

    configure_logging()

    async def _run() -> dict[str, object]:
        from sqlalchemy.ext.asyncio import async_sessionmaker

        from llm_wiki.agents.tagger import classify_case_tags, gather_case_text
        from llm_wiki.api.deps import _engine
        from llm_wiki.llm.client import LLMClient
        from llm_wiki.storage.metadata import CaseRecord

        factory = async_sessionmaker(bind=_engine, expire_on_commit=False)
        async with factory() as session:
            case = await session.get(CaseRecord, case_id)
            if case is None:
                return {"case_id": case_id, "status": "not_found"}
            if case.tags and not force:
                return {"case_id": case_id, "status": "already_tagged", "tags": case.tags}
            content = await gather_case_text(case, session)
            llm = LLMClient()
            try:
                tags = await classify_case_tags(case.title, content, llm, file_id=f"case-{case_id}")
            finally:
                await llm.aclose()
            case.tags = tags
            await session.commit()
            logger.info("autotag_case_done", case_id=case_id, tags=tags)
            return {"case_id": case_id, "status": "tagged", "tags": tags}

    with asyncio.Runner() as runner:
        return runner.run(_run())


@celery_app.task(name="llm_wiki.orchestrator.tasks.backfill_case_tags")
def backfill_case_tags(force: bool = False) -> dict[str, object]:
    """One-off backfill: queue auto-tagging for every case with no tags yet (or
    all cases when ``force``). Each becomes its own ``autotag_case`` task so the
    single worker chews through them one-by-one with per-case retry."""
    import asyncio

    from llm_wiki.logging_config import configure_logging

    configure_logging()

    async def _ids() -> list[str]:
        from sqlalchemy import select
        from sqlalchemy.ext.asyncio import async_sessionmaker

        from llm_wiki.api.deps import _engine
        from llm_wiki.storage.metadata import CaseRecord

        factory = async_sessionmaker(bind=_engine, expire_on_commit=False)
        async with factory() as session:
            cases = (await session.scalars(select(CaseRecord))).all()
        return [c.id for c in cases if force or not c.tags]

    with asyncio.Runner() as runner:
        ids = runner.run(_ids())
    for cid in ids:
        autotag_case.delay(cid, force)
    logger.info("backfill_case_tags_queued", count=len(ids), force=force)
    return {"queued": len(ids), "force": force}
