"""API route definitions — all endpoints under /api/v1/."""

from pathlib import Path

import structlog
from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.api.deps import get_db
from llm_wiki.api.schemas import (
    AuditRunRequest,
    AuditRunResponse,
    AuditStatusResponse,
    FileStatusResponse,
    FileUploadResponse,
    IssueResponse,
    LintRunResponse,
    StateEntry,
)
from llm_wiki.config import settings
from llm_wiki.orchestrator.tasks import celery_app, process_file_task, run_weekly_audit
from llm_wiki.storage.metadata import create_file_record, get_by_sha256, get_file_record
from llm_wiki.utils.hashing import sha256_stream
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

    Validates file type (.pdf / .md only), size (max 50 MB), and content hash
    (SHA-256) to detect duplicates.  Identical content that was already
    successfully ingested returns 200 with ``status="duplicate"`` so clients
    can display the original result without re-running the pipeline.

    Args:
        file: The uploaded file (multipart/form-data, field name ``file``).
        session: Injected async SQLAlchemy session.

    Returns:
        | HTTP 202  ``{"file_id", "task_id", "status": "queued"}``     — new upload
        | HTTP 200  ``{"file_id", "duplicate_of", "status": "duplicate"}`` — exact dupe

    Raises:
        HTTPException 400: Empty file, or unsupported extension (.pdf / .md only).
        HTTPException 413: File exceeds the 50 MB limit.
    """
    import io

    filename = file.filename or "upload"
    ext = Path(filename).suffix.lower()

    if ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(settings.allowed_extensions)}",
        )

    content = await file.read()

    if len(content) == 0:
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

    # ------------------------------------------------------------------
    # SHA-256 deduplication — check BEFORE writing to disk or DB
    # ------------------------------------------------------------------
    sha = sha256_stream(io.BytesIO(content))
    existing = await get_by_sha256(session, sha)
    if existing is not None:
        logger.info(
            "dedup_hit",
            sha256=sha[:16] + "…",
            original_file_id=existing.file_id,
            filename=filename,
        )
        return FileUploadResponse(
            file_id=existing.file_id,
            task_id=None,
            status="duplicate",
            duplicate_of=existing.file_id,
        )

    # ------------------------------------------------------------------
    # New file — persist and enqueue
    # ------------------------------------------------------------------
    file_id = new_file_id()
    dest = settings.raw_dir / f"{file_id}{ext}"
    dest.write_bytes(content)

    await create_file_record(session, file_id, filename, content_sha256=sha)

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
    response_model=LintRunResponse,
    status_code=status.HTTP_200_OK,
    summary="Run the deterministic Linter over the current wiki",
    tags=["quality"],
)
async def run_lint(
    dry_run: bool = False,
    checks: str = "dead_link,orphan_page,stale_date",
) -> LintRunResponse:
    """Run all deterministic quality checks synchronously and return findings.

    The Linter is fast (pure Python, no LLM) so this endpoint runs inline.
    No Celery task is created.

    Query params:
        dry_run: If True, return findings without writing to ``issues.md``.
        checks: Comma-separated subset of ``dead_link,orphan_page,stale_date``.
            Defaults to all three.

    Returns:
        200 with issue list and per-kind counters.
    """
    from datetime import datetime, timezone

    from llm_wiki.quality.issues_writer import upsert_section
    from llm_wiki.quality.linter import run_linter
    from llm_wiki.quality.models import IssueKind, IssueSection
    from llm_wiki.storage.index import IndexStorage

    requested = set(checks.split(",")) if checks else set()
    all_checks = {c.value for c in IssueKind if c.value in {"dead_link", "orphan_page", "stale_date"}}
    active_checks = requested & all_checks if requested else all_checks

    wiki_dir = settings.wiki_dir
    wiki_pages: dict[str, str] = {}
    if wiki_dir.exists():
        for md_file in sorted(wiki_dir.glob("*.md")):
            wiki_pages[md_file.stem] = md_file.read_text(encoding="utf-8")

    index_storage = IndexStorage(settings.index_path)
    headings = index_storage.read_headings()
    index_root_sections: set[str] = {
        h.section.lower().replace(" ", "-") for h in headings
    }

    current_year = datetime.now(timezone.utc).year
    issues = run_linter(
        wiki_pages=wiki_pages,
        index_root_sections=index_root_sections,
        current_year=current_year,
    )

    # Filter to requested checks
    issues = [i for i in issues if i.kind.value in active_checks]

    updated = False
    if not dry_run:
        upsert_section(
            issues_path=settings.issues_path,
            section=IssueSection.AUTO_DETECTED,
            issues=issues,
        )
        updated = True

    by_kind: dict[str, int] = {}
    for issue in issues:
        by_kind[issue.kind.value] = by_kind.get(issue.kind.value, 0) + 1

    return LintRunResponse(
        issues_found=len(issues),
        by_kind=by_kind,
        issues=[
            IssueResponse(
                kind=i.kind.value,
                section=i.section.value,
                page_slug=i.page_slug,
                description=i.description,
                related_slugs=list(i.related_slugs),
            )
            for i in issues
        ],
        issues_md_updated=updated,
    )


@router.post(
    "/audit/run",
    response_model=AuditRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Enqueue the LLM Auditor for semantic quality checks",
    tags=["quality"],
)
async def run_audit(body: AuditRunRequest | None = None) -> AuditRunResponse:
    """Enqueue an LLM Auditor run via Celery and return a task_id for polling.

    Args:
        body: Optional request body controlling mode, dry_run, sample, slugs.

    Returns:
        202 with ``task_id``, ``mode``, and estimated completion time.
    """
    from datetime import datetime, timedelta, timezone

    from llm_wiki.config import settings as cfg

    if body is None:
        body = AuditRunRequest()

    task = run_weekly_audit.delay(
        mode=body.mode,
        dry_run=body.dry_run,
        sample=body.sample,
        slugs=body.slugs,
    )

    # Rough cost estimate: $0.003 per page for sync, $0.0015 for batch
    wiki_dir = settings.wiki_dir
    n_pages = len(list(wiki_dir.glob("*.md"))) if wiki_dir.exists() else 0
    if body.sample:
        n_pages = min(n_pages, body.sample)
    if body.slugs:
        n_pages = len(body.slugs)
    rate = 0.0015 if body.mode == "batch" else 0.003
    estimated_cost = round(n_pages * rate, 4)

    completion_delta = timedelta(hours=24) if body.mode == "batch" else timedelta(minutes=2)
    estimated_at = datetime.now(timezone.utc) + completion_delta

    return AuditRunResponse(
        task_id=str(task.id),
        mode=body.mode,
        estimated_cost_usd=estimated_cost,
        estimated_completion_at=estimated_at,
    )


@router.get(
    "/audit/{task_id}",
    response_model=AuditStatusResponse,
    summary="Poll the status of an LLM Auditor task",
    tags=["quality"],
)
async def get_audit_status(task_id: str) -> AuditStatusResponse:
    """Return the current state of an Auditor Celery task.

    Args:
        task_id: Celery task ID returned by POST /audit/run.

    Returns:
        ``{"task_id", "status", "result"}`` — status is one of
        ``PENDING``, ``STARTED``, ``SUCCESS``, ``FAILURE``.
    """
    task_result = celery_app.AsyncResult(task_id)
    result = None
    if task_result.ready():
        try:
            result = task_result.get(timeout=1)
        except Exception:
            result = {"error": "task failed"}

    return AuditStatusResponse(
        task_id=task_id,
        status=task_result.status,
        result=result,
    )


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
