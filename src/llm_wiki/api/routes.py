"""API route definitions — all endpoints under /api/v1/."""

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

router = APIRouter()


# ---------------------------------------------------------------------------
# Stub placeholders — implemented task-by-task in subsequent sprints.
# Each endpoint raises 501 until its implementing task (LW-N) is merged.
# ---------------------------------------------------------------------------

@router.post(
    "/files",
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
    summary="Upload a PDF or Markdown file for ingestion (LW-5)",
    tags=["files"],
)
async def upload_file() -> JSONResponse:
    """Upload a file for async wiki ingestion.

    Implemented in LW-5. Returns 202 with file_id and task_id.
    """
    raise HTTPException(status_code=501, detail="Not implemented yet — see LW-5")


@router.get(
    "/files/{file_id}",
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
    summary="Get file processing status (LW-10)",
    tags=["files"],
)
async def get_file_status(file_id: str) -> JSONResponse:
    """Return processing state history and cost for a given file_id.

    Implemented in LW-10.
    """
    raise HTTPException(status_code=501, detail="Not implemented yet — see LW-10")


@router.get(
    "/wiki/{slug}",
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
    summary="Get a wiki page by slug (LW-16)",
    tags=["wiki"],
)
async def get_wiki_page(slug: str) -> JSONResponse:
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
async def run_lint() -> JSONResponse:
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
async def get_log(page: int = 1, per_page: int = 50) -> JSONResponse:
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
async def get_stats() -> JSONResponse:
    """Return aggregated stats: file counts, costs, last lint run.

    Implemented in LW-16.
    """
    raise HTTPException(status_code=501, detail="Not implemented yet — see LW-16")
