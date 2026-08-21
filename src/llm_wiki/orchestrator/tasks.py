"""Celery task definitions — thin wrappers around the pipeline."""

import asyncio
from typing import Any

import openai
import structlog
from celery import Celery
from celery.schedules import crontab
from kombu import Queue

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
    # Concurrency is set per-worker on the CLI (see docker-compose):
    #   worker-ingest    -Q ingest          --concurrency=2  (CPU-bound parsing)
    #   worker-artifacts -Q artifacts,light --concurrency=3  (LLM-bound, user waits)
    # The old worker_concurrency=1 guarded index.md file races — index.md is
    # gone (the knowledge map lives in Postgres now), so the guard is obsolete.
    # Route heavy user-facing work to separate queues so a 5-minute file
    # ingestion can never block artifact generation (and vice versa).
    task_routes={
        "llm_wiki.orchestrator.tasks.process_file": {"queue": "ingest"},
        "llm_wiki.orchestrator.tasks.generate_artifact": {"queue": "artifacts"},
        "llm_wiki.orchestrator.tasks.autotag_case": {"queue": "light"},
        "llm_wiki.orchestrator.tasks.backfill_case_tags": {"queue": "light"},
        "llm_wiki.orchestrator.tasks.run_weekly_audit": {"queue": "light"},
        "llm_wiki.orchestrator.tasks.sweep_stuck_generations": {"queue": "light"},
    },
    # Anything unrouted (new tasks) lands in "light" — visible immediately,
    # can't silently starve behind ingestion.
    task_default_queue="light",
    # Declare ALL queues so a worker started WITHOUT -Q (e.g. the k8s worker
    # deployment, whose command ops manages separately) consumes every queue by
    # default — otherwise the queue split silently strands ingest/artifacts
    # tasks in Redis until the manifest learns about -Q. Workers WITH an
    # explicit -Q (docker-compose worker / worker-artifacts) are unaffected and
    # keep their isolation.
    # Explicit exchange/routing_key per queue — identical to what
    # task_create_missing_queues auto-declared before this list existed
    # (a bare Queue("x") would inherit the default "light" exchange and all
    # three queues would collapse onto one binding).
    # ORDER MATTERS (queue_order_strategy=priority): user-facing first.
    task_queues=(
        Queue("artifacts", exchange="artifacts", routing_key="artifacts"),
        Queue("light", exchange="light", routing_key="light"),
        Queue("ingest", exchange="ingest", routing_key="ingest"),
        # Legacy drain: tasks enqueued BEFORE the queue split landed on an env
        # (e.g. a file uploaded minutes before the deploy) sit in the old
        # default "celery" queue — keep consuming it so that tail completes
        # instead of spinning forever. Harmless once drained (stays empty).
        Queue("celery", exchange="celery", routing_key="celery"),
    ),
    # At-least-once delivery: ack only AFTER the task finishes, and re-queue if
    # the worker dies mid-task (OOM / SIGKILL / eviction). Combined with the
    # persisted state_history the pipeline resumes instead of duplicating work.
    # ⚠️ reject_on_worker_lost re-queues WITHOUT counting toward max_retries —
    # a task that deterministically OOMs would loop forever. Every acks_late
    # task therefore needs its own delivery cap (see _bump_ingest_attempts);
    # LLM-heavy tasks opt OUT per-task (acks_late=False) — cheaper to re-click
    # than to re-bill a poison task in a kill loop.
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # With acks_late, keep at most one un-acked message per worker so a crash
    # re-queues a single task, not a whole prefetched batch.
    worker_prefetch_multiplier=1,
    # Redis re-delivers an un-acked message after this many seconds. Must exceed
    # the longest possible pipeline run so a slow (not dead) task is not
    # re-queued while still processing.
    # queue_order_strategy=priority: a worker consuming SEVERAL queues (the
    # prod deployment has no -Q) polls them in task_queues order — Redis BRPOP
    # pops the first non-empty list. Artifacts/light therefore jump ahead of an
    # ingest backlog: a user watching the «генерируется…» spinner never waits
    # behind 30 queued PDFs. Ops declined split worker deployments (replicas
    # only), so this ordering is what keeps user-facing latency sane.
    broker_transport_options={
        "visibility_timeout": 3600,
        "queue_order_strategy": "priority",
    },
    # Nobody reads Celery results (the API polls Postgres) — don't write them
    # to Redis at all. During the 2026-08-20 incident the result backend was
    # one more thing filling Redis while the queue looped.
    task_ignore_result=True,
    result_expires=3600,
    # Prefork children are recycled after N tasks or when RSS exceeds the cap
    # (in KiB) — a slow leak (httpx pools, base64 image blobs, OCR buffers)
    # then costs one child restart instead of a container OOM-kill at the k8s
    # memory limit, which with reject_on_worker_lost would re-queue the task
    # and start the kill loop again.
    worker_max_tasks_per_child=16,
    worker_max_memory_per_child=350_000,
    # Any unrouted future task gets a bounded runtime by default — nothing in
    # this app may hold a worker slot for more than 10 minutes.
    task_soft_time_limit=540,
    task_time_limit=600,
    broker_connection_retry_on_startup=True,
    beat_schedule={
        # LW-15: weekly semantic audit via OpenAI Batch API (-50% cost, 24 h SLA)
        "weekly-audit": {
            "task": "llm_wiki.orchestrator.tasks.run_weekly_audit",
            "schedule": crontab(minute=0, hour=3, day_of_week=1),  # Mon 03:00 UTC
        },
        # Janitor: no artifact/file may show «генерируется» forever. Whatever
        # kills a worker (OOM, eviction, hard limit), the stale pending rows
        # are swept to failed within 10 minutes and the UI unblocks.
        "sweep-stuck-generations": {
            "task": "llm_wiki.orchestrator.tasks.sweep_stuck_generations",
            "schedule": 600.0,
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

# Wall-clock budget for one artifact generation, BELOW the Celery soft limit:
# the asyncio.wait_for cancel fires first and the task fails cleanly (artifact
# marked failed, message acked) instead of the 2026-08-20 pattern — hard-limit
# SIGKILL of the pool child with the artifact stuck in «pending».
ARTIFACT_DEADLINE_S = 480


def _worker_session_factory():  # noqa: ANN202 — sessionmaker type is unwieldy
    """Per-call async engine for Celery tasks (NullPool).

    Each prefork child runs every task in its own short-lived event loop
    (``asyncio.Runner``). A pooled engine would hand loop-bound connections
    from a finished task's dead loop to the next task; NullPool opens and
    really closes a connection per checkout, so nothing crosses loops. The
    per-task connect cost is noise next to the LLM calls these tasks make.
    """
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    engine = create_async_engine(settings.database_url, echo=False, poolclass=NullPool)
    return async_sessionmaker(bind=engine, expire_on_commit=False)


@celery_app.task(
    bind=True,
    max_retries=3,
    acks_late=True,
    reject_on_worker_lost=True,
    name="llm_wiki.orchestrator.tasks.process_file",
    # A single file (big PDF + OCR + wiki generation) can legitimately take
    # minutes; anything past 25 min counts as hung.
    soft_time_limit=1500,
    time_limit=1800,
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

    # Poison-pill guard: reject_on_worker_lost re-queues a task whose worker
    # died (OOM / hard limit / eviction) without counting toward max_retries.
    # A file that keeps killing workers must fail, not loop forever.
    attempts = _bump_ingest_attempts(file_id)
    if attempts > INGEST_MAX_DELIVERIES:
        msg = (
            f"Обработка прерывалась {attempts - 1} раза подряд (воркер падал — "
            "вероятно, файлу не хватает памяти или времени). Файл снят с очереди."
        )
        _mark_file_failed_sync(file_id, msg)
        log.error("ingest_delivery_cap_hit", attempts=attempts)
        structlog.contextvars.clear_contextvars()
        return

    log.info("task_started", delivery_attempt=attempts)

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

        # Clean finish — zero the delivery counter so a manual re-process of
        # this file later starts with a fresh poison-pill budget.
        try:
            from sqlalchemy import create_engine, text

            _eng = create_engine(settings.database_url)
            with _eng.begin() as conn:
                conn.execute(
                    text("UPDATE files SET ingest_attempts = 0 WHERE file_id = :fid"),
                    {"fid": file_id},
                )
            _eng.dispose()
        except Exception:  # noqa: BLE001 — cosmetic, never fail a done pipeline
            pass

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


@celery_app.task(
    name="llm_wiki.orchestrator.tasks.run_weekly_audit",
    soft_time_limit=900,
    time_limit=960,
)
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


# Broker re-deliveries of an acks_late task (reject_on_worker_lost) bypass
# Celery's max_retries entirely — this DB-side cap is the only thing standing
# between a deterministic OOM and an infinite kill loop. Sized to admit the
# legitimate path (1 original + 3 self.retry() re-publishes, each of which
# also bumps the counter) plus one crash re-delivery.
INGEST_MAX_DELIVERIES = 5


def _bump_ingest_attempts(file_id: str) -> int:
    """Atomically increment and return the delivery counter for a file.

    Sync on purpose: runs before the task's event loop exists, and must work
    even when the previous delivery died so hard the async stack never ran.
    Fail-open: if the counter can't be read (DB blip), the file still gets
    processed — the guard must never become its own failure mode.
    """
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(settings.database_url)
        try:
            with engine.begin() as conn:
                row = conn.execute(
                    text(
                        "UPDATE files SET ingest_attempts = COALESCE(ingest_attempts, 0) + 1 "
                        "WHERE file_id = :fid RETURNING ingest_attempts"
                    ),
                    {"fid": file_id},
                ).first()
            return int(row[0]) if row else 1
        finally:
            engine.dispose()
    except Exception as exc:  # noqa: BLE001
        logger.warning("ingest_attempts_bump_failed", file_id=file_id, error=str(exc))
        return 1


def _mark_file_failed_sync(file_id: str, error: str) -> None:
    from sqlalchemy import create_engine, text

    engine = create_engine(settings.database_url)
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE files SET status = 'FAILED', error = :err "
                    "WHERE file_id = :fid"
                ),
                {"err": error[:500], "fid": file_id},
            )
    finally:
        engine.dispose()


def _mark_artifact_failed_sync(artifact_id: str, error: str) -> None:
    """Last-resort failure marker that works even when the task's event loop
    is already broken (SoftTimeLimitExceeded lands mid-await). Plain sync
    SQLAlchemy over the same psycopg3 URL — no loop involved."""
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(settings.database_url, poolclass=None)
        with engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE artifacts SET status='failed', error=:err, "
                    "finished_at=now() WHERE artifact_id=:aid AND status='pending'"
                ),
                {"err": error[:500], "aid": artifact_id},
            )
        engine.dispose()
    except Exception as exc:  # noqa: BLE001 — janitor will sweep it instead
        logger.warning("mark_failed_sync_failed", artifact_id=artifact_id, error=str(exc))


@celery_app.task(
    name="llm_wiki.orchestrator.tasks.generate_artifact",
    # Heavy LLM work must NOT be re-delivered by the broker: a task that kills
    # the worker (OOM / hard limit) would loop forever, re-billing the LLM on
    # every lap — exactly the 2026-08-20 prod incident. Ack up front; if the
    # worker dies, the artifact stays «pending» and the janitor fails it within
    # 10 minutes. The user re-clicks — that's the retry.
    acks_late=False,
    reject_on_worker_lost=False,
    # The internal deadline (480 s) fires first; these are the safety net.
    soft_time_limit=540,
    time_limit=600,
)
def generate_artifact(
    artifact_id: str, document_id: str, kind: str, language: str
) -> dict[str, object]:
    """Generate a heavy studio artifact in the background and store it.

    The API creates a ``pending`` artifact and enqueues this task, then the
    client polls ``GET /artifacts/{id}`` until ``status`` is ``ready``/``failed``.
    Moving the slow LLM/image work off the request path means no proxy timeout
    can drop it mid-generation.

    Every exit path leaves the artifact in a terminal state (``ready`` or
    ``failed``) — «вечно генерируется» is what the janitor exists to prevent,
    but this task tries hard to never need it.
    """
    from billiard.exceptions import SoftTimeLimitExceeded

    from llm_wiki.logging_config import configure_logging

    configure_logging()
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        task_name="generate_artifact", artifact_id=artifact_id, kind=kind
    )

    async def _run() -> dict[str, object]:
        from llm_wiki.agents.artifacts import generate_content
        from llm_wiki.llm.client import LLMClient
        from llm_wiki.storage import artifacts_store
        from llm_wiki.storage.metadata import ArtifactRecord

        factory = _worker_session_factory()
        async with factory() as session:
            record = await session.get(ArtifactRecord, artifact_id)
            # Duplicate / stale delivery guard: the row may already be resolved
            # (second click raced the first, janitor failed it, task expired in
            # queue past its usefulness). Don't burn LLM budget on it.
            if record is None or record.status != "pending":
                logger.info(
                    "generate_artifact_skipped",
                    status=getattr(record, "status", "missing"),
                )
                return {"artifact_id": artifact_id, "status": "skipped"}

            # Ops monitoring: реальный момент старта генерации воркером.
            await artifacts_store.mark_started(session, artifact_id)
            llm = LLMClient()
            try:
                content = await asyncio.wait_for(
                    generate_content(
                        session, llm, kind=kind, document_id=document_id, language=language
                    ),
                    timeout=ARTIFACT_DEADLINE_S,
                )
            except asyncio.TimeoutError:
                msg = (
                    f"Превышено время генерации ({ARTIFACT_DEADLINE_S // 60} мин) — "
                    "попробуйте ещё раз; если повторится, материал слишком объёмный"
                )
                await artifacts_store.mark_failed(session, artifact_id, msg)
                logger.warning("generate_artifact_deadline", deadline_s=ARTIFACT_DEADLINE_S)
                return {"artifact_id": artifact_id, "status": "failed"}
            except Exception as exc:  # noqa: BLE001
                await artifacts_store.mark_failed(session, artifact_id, str(exc))
                logger.warning("generate_artifact_failed", error=str(exc))
                return {"artifact_id": artifact_id, "status": "failed"}
            finally:
                await llm.aclose()
            await artifacts_store.upsert_artifact(
                session, document_id=document_id, kind=kind, language=language, content=content
            )
            logger.info("generate_artifact_done")
            return {"artifact_id": artifact_id, "status": "ready"}

    try:
        with asyncio.Runner() as runner:
            return runner.run(_run())
    except SoftTimeLimitExceeded:
        # Celery's soft limit landed somewhere the async code couldn't catch it
        # (e.g. inside DB teardown). The loop may be unusable — mark failure
        # over a plain sync connection so the UI unblocks immediately.
        _mark_artifact_failed_sync(
            artifact_id, "Отменено по таймауту воркера — попробуйте ещё раз"
        )
        logger.warning("generate_artifact_soft_limit")
        return {"artifact_id": artifact_id, "status": "failed"}
    finally:
        structlog.contextvars.clear_contextvars()


@celery_app.task(
    name="llm_wiki.orchestrator.tasks.autotag_case",
    soft_time_limit=120,
    time_limit=180,
)
def autotag_case(case_id: str, force: bool = False) -> dict[str, object]:
    """Classify a case against the fixed taxonomy and set its tags.

    Skips a case that already has tags (unless ``force``) so it never clobbers a
    user's manual edits. Best-effort — any failure leaves the tags untouched.
    """
    import asyncio

    from llm_wiki.logging_config import configure_logging

    configure_logging()

    async def _run() -> dict[str, object]:
        from llm_wiki.agents.tagger import classify_case_tags, gather_case_text
        from llm_wiki.llm.client import LLMClient
        from llm_wiki.storage.metadata import CaseRecord

        factory = _worker_session_factory()
        async with factory() as session:
            case = await session.get(CaseRecord, case_id)
            if case is None:
                return {"case_id": case_id, "status": "not_found"}
            if case.tags and case.description and not force:
                return {"case_id": case_id, "status": "already_tagged", "tags": case.tags}
            content = await gather_case_text(case, session)
            llm = LLMClient()
            try:
                tags, description = await classify_case_tags(
                    case.title, content, llm, file_id=f"case-{case_id}"
                )
            finally:
                await llm.aclose()
            # Не затираем ручные правки: теги — только если их не было (или
            # force); описание — если пустое (сбрасывается при смене состава).
            if force or not case.tags:
                case.tags = tags
            if force or not case.description:
                case.description = description
            await session.commit()
            logger.info("autotag_case_done", case_id=case_id, tags=case.tags,
                        description_len=len(case.description))
            return {"case_id": case_id, "status": "tagged", "tags": case.tags}

    with asyncio.Runner() as runner:
        return runner.run(_run())


@celery_app.task(
    name="llm_wiki.orchestrator.tasks.backfill_case_tags",
    soft_time_limit=600,
    time_limit=660,
)
def backfill_case_tags(force: bool = False) -> dict[str, object]:
    """One-off backfill: queue auto-tagging for every case with no tags yet (or
    all cases when ``force``). Each becomes its own ``autotag_case`` task so the
    single worker chews through them one-by-one with per-case retry."""
    import asyncio

    from llm_wiki.logging_config import configure_logging

    configure_logging()

    async def _ids() -> list[str]:
        from sqlalchemy import select

        from llm_wiki.storage.metadata import CaseRecord

        factory = _worker_session_factory()
        async with factory() as session:
            cases = (await session.scalars(select(CaseRecord))).all()
        return [c.id for c in cases if force or not c.tags]

    with asyncio.Runner() as runner:
        ids = runner.run(_ids())
    for cid in ids:
        autotag_case.delay(cid, force)
    logger.info("backfill_case_tags_queued", count=len(ids), force=force)
    return {"queued": len(ids), "force": force}


# Janitor windows. Started generations get the deadline plus a margin; queued
# ones get longer (a busy-but-alive artifacts queue must not be swept).
SWEEP_STARTED_AFTER_S = ARTIFACT_DEADLINE_S + 240   # 12 min after worker start
SWEEP_QUEUED_AFTER_S = 1800                          # 30 min never picked up
SWEEP_FILES_AFTER_S = 3 * 3600                       # ingest has retries/backoff


@celery_app.task(
    name="llm_wiki.orchestrator.tasks.sweep_stuck_generations",
    soft_time_limit=60,
    time_limit=90,
)
def sweep_stuck_generations() -> dict[str, int]:
    """Beat janitor: no generation may look «в работе» forever.

    Whatever killed a worker (container OOM, node eviction, hard time limit),
    the leftover ``pending`` artifacts and stuck files are moved to ``failed``
    with a human-readable reason — the UI unblocks and the user can retry,
    instead of paging ops about a «зависшая очередь».
    """
    from llm_wiki.logging_config import configure_logging

    configure_logging()

    async def _run() -> dict[str, int]:
        from datetime import datetime, timedelta, timezone

        from sqlalchemy import select

        from llm_wiki.storage.metadata import ArtifactRecord, FileRecord

        now = datetime.now(timezone.utc)
        swept_artifacts = 0
        swept_files = 0

        factory = _worker_session_factory()
        async with factory() as session:
            pending = (
                await session.scalars(
                    select(ArtifactRecord).where(ArtifactRecord.status == "pending")
                )
            ).all()
            for art in pending:
                started = art.started_at
                queued_ts = art.updated_at or art.created_at
                if started is not None:
                    if started.tzinfo is None:
                        started = started.replace(tzinfo=timezone.utc)
                    stale = now - started > timedelta(seconds=SWEEP_STARTED_AFTER_S)
                    reason = (
                        "Генерация прервана (воркер был перезапущен) — "
                        "попробуйте ещё раз"
                    )
                else:
                    if queued_ts is not None and queued_ts.tzinfo is None:
                        queued_ts = queued_ts.replace(tzinfo=timezone.utc)
                    stale = queued_ts is None or (
                        now - queued_ts > timedelta(seconds=SWEEP_QUEUED_AFTER_S)
                    )
                    reason = (
                        "Задача не дождалась воркера и была снята — "
                        "попробуйте ещё раз"
                    )
                if stale:
                    art.status = "failed"
                    art.error = reason
                    art.finished_at = now
                    swept_artifacts += 1

            active_files = (
                await session.scalars(
                    select(FileRecord).where(
                        FileRecord.status.in_(
                            [
                                "RECEIVED", "STORED", "SEARCHED", "WRITTEN",
                                "LINTED", "LOGGED", "PROCESSING", "PENDING",
                                "received", "stored", "searched", "written",
                                "linted", "logged", "processing", "pending",
                            ]
                        )
                    )
                )
            ).all()
            for fr in active_files:
                created = fr.created_at
                if created is not None and created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                if created is None or (
                    now - created > timedelta(seconds=SWEEP_FILES_AFTER_S)
                ):
                    fr.status = "FAILED"
                    fr.error = (
                        "Обработка зависла и была снята автоматически — "
                        "загрузите файл повторно"
                    )
                    swept_files += 1

            if swept_artifacts or swept_files:
                await session.commit()

        return {"artifacts_swept": swept_artifacts, "files_swept": swept_files}

    with asyncio.Runner() as runner:
        result = runner.run(_run())
    if result["artifacts_swept"] or result["files_swept"]:
        logger.warning("sweep_stuck_generations_done", **result)
    return result
