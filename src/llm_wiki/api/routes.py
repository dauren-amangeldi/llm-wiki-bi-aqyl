"""API route definitions — all endpoints under /api/v1/."""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import PlainTextResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.api.deps import check_ask_rate_limit, check_files_rate_limit, get_db
from llm_wiki.api.schemas import (
    AskRequest,
    AskResponse,
    AskSource,
    AuditRunRequest,
    AuditRunResponse,
    AuditStatusResponse,
    FileStatusResponse,
    FileUploadResponse,
    IssueResponse,
    LintRunResponse,
    LogResponse,
    StateEntry,
    StatsResponse,
    WikiPageResponse,
)
from llm_wiki.config import settings
from llm_wiki.orchestrator.tasks import celery_app, process_file_task, run_weekly_audit
from llm_wiki.storage.metadata import FileRecord, create_file_record, get_by_sha256, get_file_record
from llm_wiki.utils.backlinks import extract_backlinks
from llm_wiki.utils.hashing import sha256_stream
from llm_wiki.utils.ids import new_file_id

logger = structlog.get_logger(__name__)

router = APIRouter()

# Compiled once at module load — reused for every /log request.
_LOG_HEADER_RE = re.compile(
    r"^## (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z) — (.+)$",
    re.MULTILINE,
)

# Slug validation: lowercase alphanumeric with hyphens, min 2 chars.
# Consistent with _WIKI_LINK_RE in utils/backlinks.py.
_VALID_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]*[a-z0-9]$")


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
    _rate_check: None = Depends(check_files_rate_limit),
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
    # Kill switch — returns 503 immediately without consuming any resources.
    if not settings.ingestion_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ingestion is currently disabled (INGESTION_ENABLED=false).",
        )

    # Daily budget check — refuse early if today's spend already exceeds limits.
    # Only activated when the limits are explicitly configured as numeric values.
    _cost_limit = settings.daily_cost_limit_usd
    _token_limit = settings.daily_token_limit
    if isinstance(_cost_limit, (int, float)) or isinstance(_token_limit, int):
        from llm_wiki.quality.budget import BudgetExceeded, BudgetGuard

        guard = BudgetGuard(
            usage_log_path=settings.usage_log_path,
            daily_cost_limit_usd=_cost_limit if isinstance(_cost_limit, (int, float)) else None,
            daily_token_limit=_token_limit if isinstance(_token_limit, int) else None,
        )
        try:
            guard.check()
        except BudgetExceeded as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Daily budget exceeded — ingestion paused. {exc}",
            ) from exc

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
    summary="Get a wiki page by slug",
    tags=["wiki"],
    response_model=None,
    responses={
        200: {"description": "Wiki page in JSON or markdown form"},
        400: {"description": "Invalid slug"},
        404: {"description": "Page not found"},
    },
)
async def get_wiki_page(
    slug: str,
    request: Request,
    session: AsyncSession = Depends(get_db),
) -> WikiPageResponse | PlainTextResponse:
    """Return a wiki page as markdown or JSON depending on the ``Accept`` header.

    Args:
        slug: Page slug (e.g. ``transformers``).  Must be lowercase alphanumeric
            with hyphens, minimum 2 characters.
        request: FastAPI request — used to inspect the ``Accept`` header.
        session: Injected async SQLAlchemy session.

    Returns:
        | ``text/markdown`` if ``Accept: text/markdown`` (and JSON not preferred)
        | ``application/json`` (``WikiPageResponse``) otherwise.

    Raises:
        HTTPException 400: Slug fails validation (path traversal attempt or
            invalid characters).
        HTTPException 404: No ``.md`` file exists for the slug.
    """
    # Guard against path traversal and invalid slugs
    if ".." in slug or "/" in slug or "\\" in slug or not _VALID_SLUG_RE.match(slug):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid slug {slug!r}. Must be lowercase alphanumeric with hyphens (min 2 chars).",
        )

    page_path = settings.wiki_dir / f"{slug}.md"
    if not page_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Wiki page {slug!r} not found.",
        )

    content = page_path.read_text(encoding="utf-8")

    # Content negotiation: prefer text/markdown only when explicitly requested
    # and application/json is not also present (v1 simple substring check).
    accept = request.headers.get("accept", "")
    if "text/markdown" in accept and "application/json" not in accept:
        return PlainTextResponse(content, media_type="text/markdown; charset=utf-8")

    # Extract title from first "# Heading" line; fall back to slug.
    title = slug.replace("-", " ").title()
    for line in content.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            break

    backlinks = extract_backlinks(content)
    last_updated = datetime.fromtimestamp(page_path.stat().st_mtime, tz=timezone.utc)

    # Identify source files: FileRecord rows whose created_pages or updated_pages
    # contain this slug.  The dataset is small (<1 000 rows) so a Python-level
    # filter over a full table scan is acceptable.
    result = await session.execute(select(FileRecord).order_by(FileRecord.created_at))
    records = result.scalars().all()
    source_files = [
        r.file_id
        for r in records
        if slug in (r.created_pages or []) or slug in (r.updated_pages or [])
    ]

    return WikiPageResponse(
        slug=slug,
        title=title,
        content=content,
        backlinks=backlinks,
        last_updated=last_updated,
        source_files=source_files,
    )


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
    response_model=LogResponse,
    summary="Get ingestion changelog with pagination",
    tags=["log"],
)
async def get_log(
    page: int = Query(1, ge=1, description="1-based page number"),
    per_page: int = Query(50, ge=1, le=200, description="Entries per page (max 200)"),
) -> LogResponse:
    """Return paginated entries from ``log.md``, newest first.

    Each entry is a Markdown block starting with a ``## TIMESTAMP — filename``
    header and running until the next such header or EOF.

    Args:
        page: 1-based page number.
        per_page: Number of entries per page (1–200).

    Returns:
        ``LogResponse`` with ``total`` count and ``entries`` slice for the
        requested page.
    """
    if not settings.log_path.exists():
        return LogResponse(page=page, per_page=per_page, total=0, entries=[])

    content = settings.log_path.read_text(encoding="utf-8")
    matches = list(_LOG_HEADER_RE.finditer(content))
    if not matches:
        return LogResponse(page=page, per_page=per_page, total=0, entries=[])

    # Slice the raw text between consecutive header positions so each entry
    # includes its own header line and all body lines up to the next header.
    raw_entries: list[str] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        raw_entries.append(content[start:end].rstrip())

    raw_entries.reverse()  # newest first (log.md is append-only, chronological)

    total = len(raw_entries)
    offset = (page - 1) * per_page
    page_entries = raw_entries[offset : offset + per_page]

    return LogResponse(page=page, per_page=per_page, total=total, entries=page_entries)


@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="Get usage statistics and costs",
    tags=["stats"],
)
async def get_stats(session: AsyncSession = Depends(get_db)) -> StatsResponse:
    """Return aggregated statistics: file counts, LLM costs, and last lint run.

    Reads the following sources:
    - SQLite ``files`` table for file counts and average ingestion cost.
    - ``data/wiki/*.md`` for wiki page count.
    - ``data/usage.log`` (JSONL) for today's and this month's LLM spend.
    - ``data/issues.md`` mtime for the last Linter run timestamp.

    Args:
        session: Injected async SQLAlchemy session.

    Returns:
        ``StatsResponse`` with all aggregated values.
    """
    now = datetime.now(timezone.utc)

    # --- DB aggregates --------------------------------------------------------
    total_files_result = await session.execute(select(func.count(FileRecord.file_id)))
    total_files: int = total_files_result.scalar_one() or 0

    avg_cost_result = await session.execute(
        select(func.avg(FileRecord.cost_usd)).where(FileRecord.status == "DONE")
    )
    avg_cost_raw = avg_cost_result.scalar_one()
    avg_cost = round(float(avg_cost_raw), 4) if avg_cost_raw is not None else 0.0

    # --- Filesystem counts ----------------------------------------------------
    total_wiki_pages = (
        len(list(settings.wiki_dir.glob("*.md"))) if settings.wiki_dir.exists() else 0
    )

    # --- Usage log: cost aggregation ------------------------------------------
    cost_today = 0.0
    cost_this_month = 0.0
    today = now.date()

    if settings.usage_log_path.exists():
        for raw_line in settings.usage_log_path.read_text(encoding="utf-8").splitlines():
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                record = json.loads(raw_line)
                ts = datetime.fromisoformat(record["timestamp"])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                cost = float(record.get("cost_usd", 0.0))
                if ts.date() == today:
                    cost_today += cost
                if (ts.year, ts.month) == (now.year, now.month):
                    cost_this_month += cost
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                logger.warning("usage_log_parse_error", line=raw_line[:120])

    # --- Last lint run --------------------------------------------------------
    last_lint_run: datetime | None = None
    if settings.issues_path.exists():
        last_lint_run = datetime.fromtimestamp(
            settings.issues_path.stat().st_mtime, tz=timezone.utc
        )

    # --- Budget percentage (LW-19) -------------------------------------------
    budget_limit = settings.daily_cost_limit_usd
    budget_pct: float | None = None
    if budget_limit is not None and budget_limit > 0:
        budget_pct = round(cost_today / budget_limit * 100, 1)

    return StatsResponse(
        total_files=total_files,
        total_wiki_pages=total_wiki_pages,
        cost_today_usd=round(cost_today, 4),
        cost_this_month_usd=round(cost_this_month, 4),
        avg_cost_per_ingestion_usd=avg_cost,
        last_lint_run=last_lint_run,
        budget_cost_limit_usd=budget_limit,
        budget_cost_used_pct=budget_pct,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Q&A
# ─────────────────────────────────────────────────────────────────────────────


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question against the wiki",
    tags=["ask"],
)
async def ask_question(
    body: AskRequest,
    _rate_check: None = Depends(check_ask_rate_limit),
) -> AskResponse:
    """Run retrieval + LLM synthesis and return a cited answer.

    Stateless — each request spawns its own ``LLMClient`` so connections are
    not leaked across the event loop.  For a high-throughput server, refactor
    to a shared client behind a dependency injector later.
    """
    from llm_wiki.agents.answer import AnswerAgent
    from llm_wiki.llm.chunk_store import ChunkStore
    from llm_wiki.llm.client import LLMClient
    from llm_wiki.llm.embeddings import EmbeddingStore

    llm = LLMClient()
    try:
        embedding_store = EmbeddingStore(chroma_path=settings.chroma_dir, llm_client=llm)
        chunk_store = ChunkStore(chroma_path=settings.chroma_dir, llm_client=llm)
        agent = AnswerAgent(llm, embedding_store, chunk_store=chunk_store)
        result = await agent.answer(question=body.question, top_k=body.top_k)
    finally:
        await llm.aclose()

    return AskResponse(
        question=body.question,
        answer=result.answer,
        confidence=result.confidence,
        sources=[
            AskSource(slug=h.slug, title=h.title, similarity=h.similarity)
            for h in result.sources
        ],
        cost_usd=round(result.cost_usd, 6),
    )
