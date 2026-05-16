"""FastAPI application entry point."""

import structlog
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from llm_wiki.api.routes import router
from llm_wiki.config import settings

logger = structlog.get_logger(__name__)

app = FastAPI(
    title="LLM Wiki",
    description="LLM-powered wiki ingestion: upload PDF/MD → agents synthesize wiki pages.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.include_router(router, prefix="/api/v1")


@app.get("/health", tags=["system"])
async def health_check() -> JSONResponse:
    """Liveness probe — returns 200 when the service is up."""
    logger.info("health_check", service=settings.service_name)
    return JSONResponse({"status": "ok", "service": settings.service_name})
