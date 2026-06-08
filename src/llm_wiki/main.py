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
from llm_wiki.storage.metadata import Base, run_schema_migrations

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Create DB tables and data directories on startup."""
    # Configure structured JSON logging before anything else runs.
    configure_logging()

    # Ensure all data directories exist before any request comes in
    ensure_dirs(settings.raw_dir, settings.wiki_dir, settings.chroma_dir)

    async with _engine.begin() as conn:
        # Create all tables for a fresh database
        await conn.run_sync(Base.metadata.create_all)
        # Apply backward-compatible column additions to existing databases
        await run_schema_migrations(conn)

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

app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")
app.include_router(v1_router, prefix="/api/v1")


@app.get("/health", tags=["system"])
async def health_check() -> JSONResponse:
    """Liveness probe — returns 200 when the service is up."""
    logger.info("health_check", service=settings.service_name)
    return JSONResponse({"status": "ok", "service": settings.service_name})
