"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from llm_wiki.api.deps import _engine
from llm_wiki.api.middleware import RequestIDMiddleware
from llm_wiki.api.routes import router
from llm_wiki.api.v1 import router as v1_router
from llm_wiki.config import settings
from llm_wiki.logging_config import configure_logging
from llm_wiki.storage.filesystem import ensure_dirs
from llm_wiki.storage.metadata import Base
from llm_wiki.storage.wiki_fts import ensure_wiki_fts_table, rebuild_wiki_fts_from_store, wiki_fts_count

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Create DB tables and data directories on startup."""
    # Configure structured JSON logging before anything else runs.
    configure_logging()

    # Ensure local data directories exist (no-op effect for S3-backed raw/wiki;
    # chroma is still local until phase 3).
    ensure_dirs(settings.raw_dir, settings.wiki_dir, settings.chroma_dir)

    async with _engine.begin() as conn:
        # Create all tables for a fresh database
        await conn.run_sync(Base.metadata.create_all)
        await ensure_wiki_fts_table(conn)

    # Backfill FTS from disk when the index is empty but wiki pages exist
    from llm_wiki.api.deps import _SessionLocal

    async with _SessionLocal() as session:
        if await wiki_fts_count(session) == 0:
            indexed = await rebuild_wiki_fts_from_store(session)
            if indexed:
                logger.info("wiki_fts_startup_backfill", pages=indexed)

        from llm_wiki.storage.metadata import seed_skills, skills_count

        if await skills_count(session) == 0:
            inserted = await seed_skills(session)
            if inserted:
                logger.info("skills_startup_seed", inserted=inserted)

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestIDMiddleware)
app.include_router(router, prefix="/api/v1")
app.include_router(v1_router, prefix="/api/v1")


@app.get("/health", tags=["system"])
async def health_check() -> JSONResponse:
    """Liveness probe — returns 200 when the service is up."""
    logger.info("health_check", service=settings.service_name)
    return JSONResponse({"status": "ok", "service": settings.service_name})
