"""FastAPI application entry point."""

import asyncio
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator, Awaitable, Callable

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from llm_wiki.api.deps import _engine
from llm_wiki.api.middleware import AuthGateMiddleware, RequestIDMiddleware
from llm_wiki.api.routes import router
from llm_wiki.api.v1 import router as v1_router
from llm_wiki.config import settings
from llm_wiki.logging_config import configure_logging
from llm_wiki.storage.filesystem import ensure_dirs
from llm_wiki.storage.metadata import Base
from llm_wiki.storage.wiki_fts import ensure_wiki_fts_table

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Create DB tables and data directories on startup."""
    # Configure structured JSON logging before anything else runs.
    configure_logging()

    # Ensure local data directories exist (no-op effect for S3-backed raw/wiki).
    ensure_dirs(settings.raw_dir, settings.wiki_dir)

    async with _engine.begin() as conn:
        # pgvector must exist before create_all builds the vector() columns.
        # On a managed DB the role needs the privilege, or a DBA pre-creates it.
        from sqlalchemy import text

        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        # Create all tables for a fresh database
        await conn.run_sync(Base.metadata.create_all)
        await ensure_wiki_fts_table(conn)

    # Seed default skills on an empty database. (Wiki pages live in wiki_fts,
    # which is the source of truth — no S3 backfill needed.)
    from llm_wiki.api.deps import _SessionLocal

    async with _SessionLocal() as session:
        from llm_wiki.storage.metadata import (
            allowed_users_count,
            seed_allowed_users,
            seed_skills,
            skills_count,
        )

        if await skills_count(session) == 0:
            inserted = await seed_skills(session)
            if inserted:
                logger.info("skills_startup_seed", inserted=inserted)

        # Seed the access whitelist (demo account) on an empty DB. Enforcement
        # only kicks in when AUTH_ENABLED; seeding is harmless either way.
        if await allowed_users_count(session) == 0:
            seeded = await seed_allowed_users(session)
            if seeded:
                logger.info("allowed_users_startup_seed", inserted=seeded)

        # Seed/refresh Twins council personas + presets from the .md files.
        # seed_twin_personas upserts, so editing a persona file and restarting
        # applies the change without a migration.
        from llm_wiki.storage.metadata import seed_twin_personas

        twins_inserted = await seed_twin_personas(session)
        if twins_inserted:
            logger.info("twin_personas_startup_seed", inserted=twins_inserted)

    logger.info("startup_complete", service=settings.service_name)
    yield
    logger.info("shutdown", service=settings.service_name)


app = FastAPI(
    title="LLM Wiki",
    description="LLM-powered wiki ingestion: upload PDF/MD → agents synthesize wiki pages.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Starlette runs middleware outermost-first in reverse registration order, so
# register inner→outer: AuthGate (its 401/403 must still receive CORS headers)
# → RequestID (logs the outcome) → CORS (outermost, sets headers on every
# response). AuthGate is a no-op unless AUTH_ENABLED.
app.add_middleware(AuthGateMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router, prefix="/api/v1")
app.include_router(v1_router, prefix="/api/v1")


@app.get("/health", tags=["system"])
@app.get("/healthz", tags=["system"])
async def health_check() -> JSONResponse:
    """Liveness probe — returns 200 when the process is up. No dependency checks.

    ``/health`` is kept for backward compatibility; ``/healthz`` is the
    Kubernetes-style alias used by liveness probes.
    """
    return JSONResponse({"status": "ok", "service": settings.service_name})


async def _check_postgres() -> None:
    from sqlalchemy import text

    async with _engine.connect() as conn:
        await conn.execute(text("SELECT 1"))


async def _check_redis() -> None:
    from redis.asyncio import Redis

    client = Redis.from_url(
        settings.redis_url, socket_connect_timeout=3, socket_timeout=3
    )
    try:
        await client.ping()
    finally:
        try:
            await client.aclose()
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass


async def _check_object_store() -> None:
    from llm_wiki.storage.object_store import get_object_store

    # Cheap connectivity probe (HEAD for a missing key on S3, path check
    # locally). Resolve the store inside the thread too, so an S3 client that
    # connects on construction never blocks the event loop.
    def _probe() -> None:
        get_object_store().exists("__readyz_probe__")

    await asyncio.to_thread(_probe)


@app.get("/readyz", tags=["system"])
async def readiness() -> JSONResponse:
    """Readiness probe — 200 only when all backing services are reachable.

    Checks Postgres, Redis and the object store concurrently and returns 503
    (with a per-dependency breakdown) if any is unavailable, so the orchestrator
    can hold traffic until the pod is truly ready. Vector search lives inside
    Postgres (pgvector), so the postgres check covers it.
    """
    checks = {
        "postgres": _check_postgres,
        "redis": _check_redis,
        "object_store": _check_object_store,
    }
    results: dict[str, str] = {}

    async def _run(name: str, fn: Callable[[], Awaitable[None]]) -> None:
        try:
            await asyncio.wait_for(fn(), timeout=5.0)
            results[name] = "ok"
        except Exception as exc:  # noqa: BLE001 - report, don't raise
            results[name] = f"error: {type(exc).__name__}: {exc}"

    await asyncio.gather(*(_run(name, fn) for name, fn in checks.items()))

    ready = all(v == "ok" for v in results.values())
    if not ready:
        logger.warning("readyz_not_ready", checks=results)
    return JSONResponse(
        {"status": "ready" if ready else "not_ready", "checks": results},
        status_code=200 if ready else 503,
    )
