"""Phase 2 — file upload endpoints.

POST /uploads   — single-file upload consumed by SourcesColumn DropZone.
                  Returns the shape expected by the frontend:
                  { document_id, title, content_type, path, status }

POST /materials/upload — batch variant; handles 1..N files in one call and
                         returns { uploaded: [...], skipped: [...] } where
                         skipped contains SHA-256 duplicates.

Both endpoints reuse the ingestion pipeline from the existing
POST /api/v1/files handler (orchestrator.tasks.process_file_task) without
duplicating any logic.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Literal

import structlog
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.api.deps import get_db
from llm_wiki.config import settings
from llm_wiki.orchestrator.tasks import process_file_task
from llm_wiki.storage.metadata import create_file_record, get_by_sha256
from llm_wiki.utils.hashing import sha256_stream
from llm_wiki.utils.ids import new_file_id

logger = structlog.get_logger(__name__)

router = APIRouter()

_EXT_TO_CT: dict[str, Literal["pdf", "markdown"]] = {
    ".pdf": "pdf",
    ".md": "markdown",
}


# ---------------------------------------------------------------------------
# Shared schema
# ---------------------------------------------------------------------------


class UploadResult(BaseModel):
    document_id: str
    title: str
    content_type: str
    path: str
    status: str  # "queued" | "duplicate"


class BatchUploadResponse(BaseModel):
    uploaded: list[UploadResult]
    skipped: list[UploadResult]


# ---------------------------------------------------------------------------
# Core logic (shared by both endpoints)
# ---------------------------------------------------------------------------


async def _ingest_one(
    file: UploadFile,
    session: AsyncSession,
) -> tuple[UploadResult, bool]:
    """Validate, dedup-check, persist, and enqueue one uploaded file.

    Returns:
        (result, was_skipped) where was_skipped=True for SHA-256 duplicates.

    Raises:
        HTTPException 400/413 for invalid files.
    """
    filename = file.filename or "upload"
    ext = Path(filename).suffix.lower()

    if ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(settings.allowed_extensions)}",
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file is not allowed.",
        )

    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {settings.max_file_size_mb} MB limit.",
        )

    sha = sha256_stream(io.BytesIO(content))
    existing = await get_by_sha256(session, sha)

    if existing is not None:
        orig_name = existing.original_name
        ext_orig = Path(orig_name).suffix.lower()
        result = UploadResult(
            document_id=existing.file_id,
            title=Path(orig_name).stem,
            content_type=_EXT_TO_CT.get(ext_orig, "markdown"),
            path=f"/api/v1/files/{existing.file_id}/raw",
            status="duplicate",
        )
        logger.info("upload_duplicate", file_id=existing.file_id, filename=filename)
        return result, True

    file_id = new_file_id()
    dest = settings.raw_dir / f"{file_id}{ext}"
    dest.write_bytes(content)
    await create_file_record(session, file_id, filename, content_sha256=sha)
    process_file_task.delay(file_id)

    title = Path(filename).stem
    result = UploadResult(
        document_id=file_id,
        title=title,
        content_type=_EXT_TO_CT.get(ext, "markdown"),
        path=f"/api/v1/files/{file_id}/raw",
        status="queued",
    )
    logger.info("upload_queued", file_id=file_id, filename=filename)
    return result, False


# ---------------------------------------------------------------------------
# POST /uploads  — single file (DropZone contract)
# ---------------------------------------------------------------------------


@router.post(
    "/uploads",
    response_model=UploadResult,
    status_code=status.HTTP_200_OK,
    tags=["upload"],
)
async def upload_single(
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_db),
) -> UploadResult:
    """Accept one file and enqueue it for ingestion.

    Returns the UploadResult regardless of duplicate status so the DropZone
    can display the source immediately.  Duplicates return the existing
    ``document_id`` with ``status="duplicate"``.
    """
    result, _ = await _ingest_one(file, session)
    return result


# ---------------------------------------------------------------------------
# POST /materials/upload  — batch (1..N files)
# ---------------------------------------------------------------------------


@router.post(
    "/materials/upload",
    response_model=BatchUploadResponse,
    status_code=status.HTTP_200_OK,
    tags=["upload"],
)
async def upload_batch(
    files: list[UploadFile] = File(...),
    session: AsyncSession = Depends(get_db),
) -> BatchUploadResponse:
    """Accept one or more files and enqueue each for ingestion.

    Duplicates (by SHA-256) are separated into ``skipped`` so the caller can
    show appropriate UI feedback without re-running the ingestion pipeline.
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one file is required.",
        )

    uploaded: list[UploadResult] = []
    skipped: list[UploadResult] = []

    for f in files:
        result, was_skipped = await _ingest_one(f, session)
        (skipped if was_skipped else uploaded).append(result)

    return BatchUploadResponse(uploaded=uploaded, skipped=skipped)
