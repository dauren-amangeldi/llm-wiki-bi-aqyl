"""API route definitions — all endpoints under /api/v1/."""

from pathlib import Path

import structlog
from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.api.deps import get_db
from llm_wiki.api.schemas import FileStatusResponse, FileUploadResponse, StateEntry
from llm_wiki.config import settings
from llm_wiki.orchestrator.tasks import process_file_task
from llm_wiki.storage.metadata import create_file_record, get_file_record
from llm_wiki.utils.ids import new_file_id

logger = structlog.get_logger(__name__)

router = APIRouter()


@router.post(
    "/files",
    response_model=FileUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a PDF or Markdown file for ingestion",
    tags=["files"],
)
async def upload_file(
    file: UploadFile,
    session: AsyncSession = Depends(get_db),
) -> FileUploadResponse:
    """Accept a PDF or Markdown file and enqueue it for async wiki ingestion.

    Validates file type (.pdf / .md only) and size (max 50 MB), saves the raw
    file to ``/raw/``, creates a ``FileRecord`` in SQLite, and dispatches a
    Celery task.

    Args:
        file: The uploaded file (multipart/form-data, field name ``file``).
        session: Injected async SQLAlchemy session.

    Returns:
        202 with ``file_id``, ``task_id``, and ``status="queued"``.

    Raises:
        HTTPException 400: Unsupported file type (not .pdf or .md).
        HTTPException 413: File exceeds the 50 MB size limit.
    """
    filename = file.filename or "upload"
    ext = Path(filename).suffix.lower()

    if ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(settings.allowed_extensions)}",
        )

    # Read content to validate size (streaming would be better at scale, fine for MVP)
    content = await file.read()
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {settings.max_file_size_mb} MB limit.",
        )

    file_id = new_file_id()
    dest = settings.raw_dir / f"{file_id}{ext}"
    dest.write_bytes(content)

    await create_file_record(session, file_id, filename)

    task = process_file_task.delay(file_id)

    logger.info("file_uploaded", file_id=file_id, filename=filename, size_bytes=len(content))

    return FileUploadResponse(file_id=file_id, task_id=str(task.id), status="queued")


# ---------------------------------------------------------------------------
# Stub placeholders — implemented task-by-task in subsequent sprints.
# ---------------------------------------------------------------------------


@router.get(
    "/files/{file_id}",
    response_model=FileStatusResponse,
    summary="Get file processing status",
    tags=["files"],
)
async def get_file_status(
    file_id: str,
    session: AsyncSession = Depends(get_db),
) -> FileStatusResponse:
    """Return processing state, state history, affected pages, and cost.

    Args:
        file_id: UUID of the file to query.
        session: Injected async SQLAlchemy session.

    Returns:
        Full status record including state transitions and cost.

    Raises:
        HTTPException 404: No file with the given ``file_id`` exists.
    """
    record = await get_file_record(session, file_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"File {file_id!r} not found")

    return FileStatusResponse(
        file_id=record.file_id,
        original_name=record.original_name,
        status=record.status,
        state_history=[
            StateEntry(state=e["state"], at=e["at"])
            for e in (record.state_history or [])
        ],
        created_pages=list(record.created_pages or []),
        updated_pages=list(record.updated_pages or []),
        cost_usd=record.cost_usd,
    )


@router.get(
    "/wiki/{slug}",
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
    summary="Get a wiki page by slug (LW-16)",
    tags=["wiki"],
)
async def get_wiki_page(slug: str) -> None:
    """Return a wiki page as markdown or JSON depending on Accept header.

    Implemented in LW-16.
    """
    raise HTTPException(status_code=501, detail="Not implemented yet — see LW-16")


@router.post(
    "/lint/run",
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
    summary="Manually trigger the Lint Agent (LW-15)",
    tags=["lint"],
)
async def run_lint() -> None:
    """Enqueue a Lint Agent run. Returns 202 with task_id.

    Implemented in LW-15.
    """
    raise HTTPException(status_code=501, detail="Not implemented yet — see LW-15")


@router.get(
    "/log",
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
    summary="Get ingestion changelog with pagination (LW-16)",
    tags=["log"],
)
async def get_log(page: int = 1, per_page: int = 50) -> None:
    """Return paginated entries from log.md.

    Implemented in LW-16.
    """
    raise HTTPException(status_code=501, detail="Not implemented yet — see LW-16")


@router.get(
    "/stats",
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
    summary="Get usage statistics and costs (LW-16)",
    tags=["stats"],
)
async def get_stats() -> None:
    """Return aggregated stats: file counts, costs, last lint run.

    Implemented in LW-16.
    """
    raise HTTPException(status_code=501, detail="Not implemented yet — see LW-16")
