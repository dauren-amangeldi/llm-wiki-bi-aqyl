"""API route definitions — all endpoints under /api/v1/."""

import json
import re
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from pathlib import Path

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import PlainTextResponse, StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.api.deps import check_ask_rate_limit, check_files_rate_limit, get_current_user, get_db
from llm_wiki.api.schemas import (
    AdvisorPointResponse,
    AdvisorRequest,
    AskRequest,
    AskResponse,
    AskSource,
    AuditRunRequest,
    AuditRunResponse,
    AuditStatusResponse,
    CurrentUser,
    FileStatusResponse,
    FileUploadResponse,
    IssueResponse,
    LintRunResponse,
    LogResponse,
    NotebookAskRequest,
    NotebookAttachRequest,
    NotebookCreateRequest,
    NotebookFileRef,
    NotebookResponse,
    SkillResponse,
    SkillUpdateRequest,
    StateEntry,
    StatsResponse,
    WikiPageResponse,
    WikiSearchResult,
)
from llm_wiki.config import settings
from llm_wiki.orchestrator.tasks import celery_app, process_file_task, run_weekly_audit
from llm_wiki.storage.metadata import FileRecord, create_file_record, get_by_sha256, get_file_record
from llm_wiki.storage.wiki_fts import keyword_search
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
# Keyword search (FTS5 — no LLM, LW-N5)
# ─────────────────────────────────────────────────────────────────────────────


@router.get(
    "/search",
    response_model=list[WikiSearchResult],
    summary="Lexical full-text search over wiki pages",
    tags=["search"],
)
async def search_wiki(
    q: str = Query("", description="Search query (min 3 characters)"),
    scope: str = Query("all", description="Scope filter (accepted, ignored for MVP)"),
    language: str = Query("ru", description="Language (accepted, ignored for MVP)"),
    limit: int = Query(10, ge=1, le=100),
    session: AsyncSession = Depends(get_db),
) -> list[WikiSearchResult]:
    """Run FTS5 keyword search over indexed wiki pages.

    No LLM calls — pure SQLite ``MATCH``.  Returns an empty list when
    ``q`` is shorter than 3 characters.
    """
    _ = scope, language  # accepted for API compat; filtering deferred
    if len(q.strip()) < 3:
        return []

    hits = await keyword_search(session, q, limit=limit)
    return [
        WikiSearchResult(
            slug=h.slug,
            title=h.title,
            snippet=h.snippet,
            scope="wiki",
        )
        for h in hits
    ]


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


# ─────────────────────────────────────────────────────────────────────────────
# Advisor (semantic SSE — LW-N8)
# ─────────────────────────────────────────────────────────────────────────────


def _sse_line(payload: dict[str, object]) -> str:
    """Format a single Server-Sent Event data line."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post(
    "/advisor",
    summary="Advisor — semantic case search with SSE streaming",
    tags=["advisor"],
    response_class=StreamingResponse,
)
async def advisor_endpoint(
    body: AdvisorRequest,
    stream: bool = Query(False, description="When true, stream SSE progress events"),
    session: AsyncSession = Depends(get_db),
    _rate_check: None = Depends(check_ask_rate_limit),
) -> StreamingResponse:
    """Run the Advisor Agent and return structured insights via SSE.

    The frontend hook ``useSSEStream`` appends ``?stream=true`` automatically.
    Progress events use ``status``; the final event has ``done: true`` plus either
    the full ``AdvisorResponse`` fields or a ``refusal`` payload.
    """
    from llm_wiki.agents.advisor import AdvisorAgent, HistoryTurn
    from llm_wiki.llm.chunk_store import ChunkStore
    from llm_wiki.llm.client import LLMClient
    from llm_wiki.storage.metadata import resolve_skill_system_prompt

    role_prompt = await resolve_skill_system_prompt(session, body.role)

    async def event_generator() -> AsyncGenerator[str, None]:
        llm = LLMClient()
        try:
            if stream:
                yield _sse_line({"status": "searching", "step": 1, "total": 2})

            chunk_store = ChunkStore(chroma_path=settings.chroma_dir, llm_client=llm)
            agent = AdvisorAgent(llm, chunk_store)
            history = [
                HistoryTurn(role=t.role, content=t.content) for t in body.history
            ]
            result = await agent.advise(
                query=body.query,
                role=body.role,
                language=body.language,
                history=history or None,
                system_prompt=role_prompt,
            )

            if stream:
                yield _sse_line({"status": "synthesizing", "step": 2, "total": 2})

            if result.refusal:
                yield _sse_line(
                    {
                        "done": True,
                        "refusal": True,
                        "refusal_message": result.refusal_message,
                    }
                )
            else:
                yield _sse_line(
                    {
                        "done": True,
                        "title": result.title,
                        "summary": result.summary,
                        "points": [
                            AdvisorPointResponse(
                                heading=p.heading,
                                body=p.body,
                                metric=p.metric,
                                tag=p.tag,
                                case_id=p.case_id,
                            ).model_dump()
                            for p in result.points
                        ],
                        "source": result.source,
                        "caseCount": result.caseCount,
                        "cost_usd": round(result.cost_usd, 6),
                    }
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("advisor_stream_failed", error=str(exc))
            yield _sse_line({"error": str(exc)})
        finally:
            await llm.aclose()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Skills — role-based prompts (LW-N12)
# ─────────────────────────────────────────────────────────────────────────────


def _skill_to_response(skill: object) -> SkillResponse:
    from llm_wiki.storage.metadata import Skill, skill_role_to_slug

    assert isinstance(skill, Skill)
    return SkillResponse(
        slug=skill_role_to_slug(skill.role),
        name=skill.title,
        content=skill.system_prompt,
        role=skill.role,
        active=bool(skill.active),
        description=skill.description,
    )


@router.get(
    "/skills",
    response_model=list[SkillResponse],
    summary="List role-based AI skills",
    tags=["skills"],
)
async def list_skills_endpoint(
    session: AsyncSession = Depends(get_db),
) -> list[SkillResponse]:
    """Return all configured skills for the management UI."""
    from llm_wiki.storage.metadata import list_skills

    rows = await list_skills(session)
    return [_skill_to_response(s) for s in rows]


@router.put(
    "/skills/{role:path}",
    response_model=SkillResponse,
    summary="Update a skill's system prompt or active flag",
    tags=["skills"],
)
async def update_skill_endpoint(
    role: str,
    body: SkillUpdateRequest,
    session: AsyncSession = Depends(get_db),
) -> SkillResponse:
    """Persist ``system_prompt`` / ``active`` for the given role."""
    from llm_wiki.storage.metadata import skill_slug_to_role, update_skill

    role_key = skill_slug_to_role(role)
    prompt = body.resolved_system_prompt()
    if prompt is None and body.active is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide content/system_prompt and/or active.",
        )

    updated = await update_skill(
        session,
        role_key,
        system_prompt=prompt,
        active=body.active,
    )
    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Skill {role_key!r} not found.",
        )
    return _skill_to_response(updated)


# ─────────────────────────────────────────────────────────────────────────────
# Notebooks — scoped sources, no global wiki writes (LW-N15 / LW-N16 / LW-N17)
# ─────────────────────────────────────────────────────────────────────────────


def _notebook_response(notebook: object, files: list[object]) -> NotebookResponse:
    from llm_wiki.storage.metadata import FileRecord, Notebook

    assert isinstance(notebook, Notebook)
    file_refs: list[NotebookFileRef] = []
    for record in files:
        assert isinstance(record, FileRecord)
        file_refs.append(
            NotebookFileRef(
                file_id=record.file_id,
                original_name=record.original_name,
                status=record.status,
            )
        )
    return NotebookResponse(
        id=notebook.id,
        title=notebook.title,
        created_at=notebook.created_at,
        updated_at=notebook.updated_at,
        files=file_refs,
    )


async def _get_owned_notebook_or_404(
    session: AsyncSession,
    notebook_id: str,
    user: CurrentUser,
) -> object:
    from llm_wiki.storage.metadata import Notebook, get_notebook

    notebook = await get_notebook(session, notebook_id, user.id)
    if notebook is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Notebook {notebook_id!r} not found.",
        )
    return notebook


@router.post(
    "/notebooks",
    response_model=NotebookResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a notebook",
    tags=["notebooks"],
)
async def create_notebook_endpoint(
    body: NotebookCreateRequest,
    session: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
) -> NotebookResponse:
    """Create an empty notebook owned by the current user."""
    from llm_wiki.storage.metadata import create_notebook

    notebook = await create_notebook(session, user.id, body.title)
    return _notebook_response(notebook, [])


@router.get(
    "/notebooks",
    response_model=list[NotebookResponse],
    summary="List notebooks for the current user",
    tags=["notebooks"],
)
async def list_notebooks_endpoint(
    session: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
) -> list[NotebookResponse]:
    """Return all notebooks owned by the caller, newest first."""
    from llm_wiki.storage.metadata import list_notebooks, notebook_files_detail

    notebooks = await list_notebooks(session, user.id)
    results: list[NotebookResponse] = []
    for nb in notebooks:
        files = await notebook_files_detail(session, nb.id, user.id)
        results.append(_notebook_response(nb, files))
    return results


@router.get(
    "/notebooks/{notebook_id}",
    response_model=NotebookResponse,
    summary="Get a notebook by id",
    tags=["notebooks"],
)
async def get_notebook_endpoint(
    notebook_id: str,
    session: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
) -> NotebookResponse:
    """Return one notebook when owned by the caller."""
    from llm_wiki.storage.metadata import notebook_files_detail

    notebook = await _get_owned_notebook_or_404(session, notebook_id, user)
    files = await notebook_files_detail(session, notebook_id, user.id)
    return _notebook_response(notebook, files)


@router.delete(
    "/notebooks/{notebook_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a notebook",
    tags=["notebooks"],
)
async def delete_notebook_endpoint(
    notebook_id: str,
    session: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
) -> None:
    """Delete a notebook and its file links (sources remain in raw/chroma)."""
    from llm_wiki.storage.metadata import delete_notebook

    deleted = await delete_notebook(session, notebook_id, user.id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Notebook {notebook_id!r} not found.",
        )


@router.post(
    "/notebooks/{notebook_id}/files",
    response_model=NotebookResponse,
    summary="Upload or attach a source file to a notebook",
    tags=["notebooks"],
)
async def add_notebook_file_endpoint(
    notebook_id: str,
    session: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
    file: UploadFile | None = File(None),
    file_id: str | None = Form(None),
) -> NotebookResponse:
    """Add a source to a notebook without running the global wiki pipeline.

    Accepts either a multipart *file* upload (parsed + embedded) or an existing
    *file_id* to attach without re-embedding.
    """
    from llm_wiki.llm.chunk_store import ChunkStore
    from llm_wiki.llm.client import LLMClient
    from llm_wiki.orchestrator.pipeline import (
        attach_notebook_existing_file,
        ingest_notebook_file,
    )
    from llm_wiki.storage.metadata import Notebook, notebook_files_detail

    await _get_owned_notebook_or_404(session, notebook_id, user)

    llm = LLMClient()
    try:
        chunk_store = ChunkStore(chroma_path=settings.chroma_dir, llm_client=llm)
        if file is not None and file.filename:
            content = await file.read()
            if len(content) > settings.max_file_size_mb * 1024 * 1024:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File exceeds {settings.max_file_size_mb} MB limit.",
                )
            try:
                await ingest_notebook_file(
                    session,
                    notebook_id,
                    user.id,
                    file.filename,
                    content,
                    llm,
                    chunk_store,
                )
            except LookupError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        elif file_id:
            try:
                await attach_notebook_existing_file(
                    session, notebook_id, user.id, file_id
                )
            except LookupError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Provide a file upload or file_id.",
            )
    finally:
        await llm.aclose()

    notebook = await _get_owned_notebook_or_404(session, notebook_id, user)
    assert isinstance(notebook, Notebook)
    files = await notebook_files_detail(session, notebook_id, user.id)
    return _notebook_response(notebook, files)


@router.post(
    "/notebooks/{notebook_id}/attach",
    response_model=NotebookResponse,
    summary="Attach an existing file to a notebook (JSON)",
    tags=["notebooks"],
)
async def attach_notebook_file_json(
    notebook_id: str,
    body: NotebookAttachRequest,
    session: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
) -> NotebookResponse:
    """JSON variant of file attach — used when linking an existing case."""
    from llm_wiki.orchestrator.pipeline import attach_notebook_existing_file
    from llm_wiki.storage.metadata import Notebook, notebook_files_detail

    await _get_owned_notebook_or_404(session, notebook_id, user)
    try:
        await attach_notebook_existing_file(
            session, notebook_id, user.id, body.file_id
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    notebook = await _get_owned_notebook_or_404(session, notebook_id, user)
    assert isinstance(notebook, Notebook)
    files = await notebook_files_detail(session, notebook_id, user.id)
    return _notebook_response(notebook, files)


@router.post(
    "/notebooks/{notebook_id}/ask",
    summary="Ask a question scoped to notebook sources (SSE)",
    tags=["notebooks"],
    response_class=StreamingResponse,
)
async def notebook_ask_endpoint(
    notebook_id: str,
    body: NotebookAskRequest,
    stream: bool = Query(False),
    session: AsyncSession = Depends(get_db),
    user: CurrentUser = Depends(get_current_user),
    _rate_check: None = Depends(check_ask_rate_limit),
) -> StreamingResponse:
    """Run scoped RAG over notebook file_ids only; stream progress + answer."""
    from llm_wiki.agents.answer import AnswerAgent
    from llm_wiki.llm.chunk_store import ChunkStore
    from llm_wiki.llm.client import LLMClient
    from llm_wiki.storage.metadata import notebook_file_ids

    await _get_owned_notebook_or_404(session, notebook_id, user)
    scoped_ids = await notebook_file_ids(session, notebook_id, user.id)

    async def event_generator() -> AsyncGenerator[str, None]:
        llm = LLMClient()
        try:
            if stream:
                yield _sse_line({"status": "searching", "step": 1, "total": 2})

            chunk_store = ChunkStore(chroma_path=settings.chroma_dir, llm_client=llm)
            from llm_wiki.llm.embeddings import EmbeddingStore

            embedding_store = EmbeddingStore(
                chroma_path=settings.chroma_dir, llm_client=llm
            )
            agent = AnswerAgent(llm, embedding_store, chunk_store=chunk_store)
            result = await agent.answer_for_notebook(
                question=body.question,
                file_ids=scoped_ids,
                language=body.language,
                file_id=f"nb-{notebook_id}",
            )

            if stream:
                yield _sse_line({"status": "synthesizing", "step": 2, "total": 2})

            refusal = result.confidence == "low" and not result.sources
            payload: dict[str, object] = {
                "done": True,
                "answer": result.answer,
                "confidence": result.confidence,
                "sources": [
                    {"slug": s.slug, "title": s.title, "similarity": s.similarity}
                    for s in result.sources
                ],
                "cost_usd": round(result.cost_usd, 6),
            }
            if refusal:
                payload["refusal"] = True
                payload["refusal_message"] = result.answer
            yield _sse_line(payload)
        except Exception as exc:  # noqa: BLE001
            logger.exception("notebook_ask_failed", error=str(exc))
            yield _sse_line({"error": str(exc)})
        finally:
            await llm.aclose()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
