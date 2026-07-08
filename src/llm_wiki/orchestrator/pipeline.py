"""Ingestion pipeline state machine.

State transitions:
    RECEIVED → STORED → SEARCHED → WRITTEN → LINTED → LOGGED → DONE
                                ↘ FAILED (with step + error)

Each step is idempotent: safe to re-run with the same file_id.
"""

import json
from enum import StrEnum
from pathlib import Path

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llm_wiki.agents.search import SearchAgent
from llm_wiki.agents.writer import WikiPage, WriterAgent
from llm_wiki.config import settings
from llm_wiki.llm.chunk_store import ChunkStore
from llm_wiki.llm.client import LLMClient
from llm_wiki.llm.embeddings import EmbeddingStore, SearchHit
from llm_wiki.parsers.markdown import parse_markdown_file
from llm_wiki.parsers.pdf import parse_pdf
from llm_wiki.storage.backlinks_sync import sync_backlinks_for_page
from llm_wiki.storage.chunk_sync import sync_chunks_for_page
from llm_wiki.storage.index import IndexStorage
from llm_wiki.utils.backlinks import extract_outgoing_links
from llm_wiki.storage.log import append_log_entry
from llm_wiki.storage.metadata import (
    FileRecord,
    append_state_history,
    get_file_record,
    update_file_status,
)

logger = structlog.get_logger(__name__)


class FileState(StrEnum):
    """All possible states of a file through the ingestion pipeline."""

    RECEIVED = "RECEIVED"
    STORED = "STORED"
    SEARCHED = "SEARCHED"
    WRITTEN = "WRITTEN"
    LINTED = "LINTED"
    LOGGED = "LOGGED"
    DONE = "DONE"
    FAILED = "FAILED"


async def process_file(file_id: str) -> None:
    """Run the full ingestion pipeline for *file_id*.

    Steps:
        1. STORED   — parse the raw file into plain text
        2. SEARCHED — Search Agent identifies relevant wiki pages
        3. WRITTEN  — Writer Agent creates or updates pages + index.md
        4. LOGGED   — append entry to log.md, record cost in DB
        5. DONE     — mark complete

    Each step is guarded by an idempotency check: if the state already exists
    in ``state_history`` the step is skipped and its results are recomputed
    so subsequent steps can proceed.

    On any unhandled exception the record is transitioned to FAILED and the
    exception is re-raised for Celery to handle retry logic.

    The LINTED step runs the deterministic Linter after WRITTEN.  A Linter
    failure **does not** fail the pipeline — quality checks are important but
    less critical than the ingestion itself.

    Args:
        file_id: UUID of the file to process (must exist in the DB).
    """
    from llm_wiki.api.deps import _engine

    session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
        bind=_engine, expire_on_commit=False, autoflush=False
    )

    async with session_factory() as session:
        record = await get_file_record(session, file_id)
        if record is None:
            raise ValueError(f"File {file_id!r} not found in database")

        completed: set[str] = {e["state"] for e in (record.state_history or [])}
        logger.info(
            "pipeline_started",
            file_id=file_id,
            status=record.status,
            completed_states=sorted(completed),
        )

        # One LLMClient for the whole pipeline run; closed in finally so its
        # httpx connections are released before the event loop shuts down.
        llm = LLMClient()
        try:
            # ----------------------------------------------------------------
            # STORED — parse raw file to plain text
            # ----------------------------------------------------------------
            logger.info("pipeline_step_start", file_id=file_id, step="STORED")
            file_text = _load_raw_text(file_id, record.raw_key)

            if "STORED" not in completed:
                await _transition(session, file_id, "STORED")
            logger.info("pipeline_step_done", file_id=file_id, step="STORED")

            # ----------------------------------------------------------------
            # SEARCHED — Search Agent v2 (embedding pre-filter + LLM re-rank)
            # ----------------------------------------------------------------
            logger.info("pipeline_step_start", file_id=file_id, step="SEARCHED")
            embedding_store = EmbeddingStore(
                llm_client=llm
            )
            chunk_store = ChunkStore(
                llm_client=llm
            )
            index_storage = IndexStorage(
                settings.index_path, embedding_store=embedding_store
            )
            # heading_texts kept for backward-compat with SearchAgent.run() signature
            headings = index_storage.read_headings()
            heading_texts = [h.text for h in headings]

            search_agent = SearchAgent(llm, embedding_store)
            search_results: list[SearchHit] = await search_agent.run(
                file_text, heading_texts, file_id=file_id
            )

            if "SEARCHED" not in completed:
                await _transition(session, file_id, "SEARCHED")
            logger.info("pipeline_step_done", file_id=file_id, step="SEARCHED")

            # ----------------------------------------------------------------
            # WRITTEN — Writer Agent creates / updates pages
            # ----------------------------------------------------------------
            if "WRITTEN" not in completed:
                logger.info("pipeline_step_start", file_id=file_id, step="WRITTEN")
                writer = WriterAgent(llm)
                created_pages: list[str] = []
                updated_pages: list[str] = []

                if not search_results:
                    # Scenario A — brand-new topic
                    page = await writer.create_page(file_text, file_id)
                    _save_wiki_page(page)
                    sync_chunks_for_page(
                        chunk_store=chunk_store,
                        slug=page.slug,
                        title=page.title,
                        content=page.content,
                        file_id=file_id,
                    )
                    index_storage.add_page(page.slug, "General", title=page.title)
                    created_pages.append(page.slug)
                    # Synchronise backlinks: new page has no previous outgoing links
                    sync_backlinks_for_page(
                        source_slug=page.slug,
                        new_content=page.content,
                        previous_outgoing=(),
                        file_id=file_id,
                    )
                else:
                    # Scenario B — update existing pages (up to 5)
                    existing = _load_existing_pages(search_results[:5])

                    if existing:
                        # Capture outgoing links BEFORE the Writer Agent rewrites pages
                        previous_outgoing_by_slug: dict[str, list[str]] = {
                            p.slug: extract_outgoing_links(p.content) for p in existing
                        }
                        pages_out = await writer.update_pages(file_text, existing, file_id)
                        for p in pages_out:
                            _save_wiki_page(p)
                            sync_chunks_for_page(
                                chunk_store=chunk_store,
                                slug=p.slug,
                                title=p.title,
                                content=p.content,
                                file_id=file_id,
                            )
                            updated_pages.append(p.slug)
                            # Synchronise backlinks using pre-write outgoing links as baseline
                            sync_backlinks_for_page(
                                source_slug=p.slug,
                                new_content=p.content,
                                previous_outgoing=previous_outgoing_by_slug.get(p.slug, []),
                                file_id=file_id,
                            )
                    else:
                        # Search found headings but files are absent — create new
                        page = await writer.create_page(file_text, file_id)
                        _save_wiki_page(page)
                        sync_chunks_for_page(
                            chunk_store=chunk_store,
                            slug=page.slug,
                            title=page.title,
                            content=page.content,
                            file_id=file_id,
                        )
                        index_storage.add_page(page.slug, "General", title=page.title)
                        created_pages.append(page.slug)
                        # Synchronise backlinks: new page has no previous outgoing links
                        sync_backlinks_for_page(
                            source_slug=page.slug,
                            new_content=page.content,
                            previous_outgoing=(),
                            file_id=file_id,
                        )

                # Persist page lists to DB
                from sqlalchemy import update as sa_update

                await session.execute(
                    sa_update(FileRecord)
                    .where(FileRecord.file_id == file_id)
                    .values(created_pages=created_pages, updated_pages=updated_pages)
                )
                await session.commit()
                await _transition(session, file_id, "WRITTEN")
                logger.info("pipeline_step_done", file_id=file_id, step="WRITTEN")

            # ----------------------------------------------------------------
            # LINTED — deterministic quality checks on the whole wiki
            # Failure here is non-fatal: wrap in try/except and continue.
            # ----------------------------------------------------------------
            if "LINTED" not in completed:
                try:
                    _run_linter_step(file_id=file_id)
                except Exception as lint_exc:  # noqa: BLE001
                    logger.error(
                        "linter_failed",
                        file_id=file_id,
                        error=str(lint_exc),
                    )
                else:
                    await _transition(session, file_id, "LINTED")

            # ----------------------------------------------------------------
            # LOGGED — append to log.md, persist cost
            # ----------------------------------------------------------------
            if "LOGGED" not in completed:
                record = await get_file_record(session, file_id)
                cost = _sum_cost(settings.usage_log_path, file_id)
                append_log_entry(
                    log_path=settings.log_path,
                    file_id=file_id,
                    original_name=record.original_name if record else "unknown",
                    created_pages=list(record.created_pages) if record else [],
                    updated_pages=list(record.updated_pages) if record else [],
                    cost_usd=cost,
                )
                from sqlalchemy import update as sa_update

                await session.execute(
                    sa_update(FileRecord)
                    .where(FileRecord.file_id == file_id)
                    .values(cost_usd=cost)
                )
                await session.commit()
                await _transition(session, file_id, "LOGGED")

            # ----------------------------------------------------------------
            # DONE
            # ----------------------------------------------------------------
            await _transition(session, file_id, "DONE")
            await update_file_status(session, file_id, "DONE")
            logger.info("pipeline_done", file_id=file_id)

        except Exception as exc:
            logger.error("pipeline_failed", file_id=file_id, error=str(exc))
            await update_file_status(session, file_id, "FAILED")
            raise
        finally:
            # Always close the SDK client within the active event loop so that
            # httpx can release its connection pool cleanly.  Without this,
            # GC would try to close the httpx.AsyncClient after the loop is
            # gone, producing "RuntimeError: Event loop is closed" warnings.
            await llm.aclose()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


async def _transition(session: AsyncSession, file_id: str, state: str) -> None:
    """Append *state* to the file's history and update its status field.

    Args:
        session: Active async SQLAlchemy session.
        file_id: UUID of the file being processed.
        state: New pipeline state (e.g. ``"STORED"``).
    """
    await append_state_history(session, file_id, state)
    await update_file_status(session, file_id, state)
    logger.info("state_transition", file_id=file_id, state=state)


def _load_raw_text(file_id: str, stored_key: str | None = None) -> str:
    """Find the uploaded source for *file_id* and extract its plain text.

    Uses the date-partitioned key persisted on the FileRecord (*stored_key*)
    when available; falls back to a legacy scan of the ``raw/`` prefix for
    files uploaded before date-partitioning.

    Raises:
        FileNotFoundError: If no raw object for *file_id* exists.
        ValueError: If the file extension is not ``.pdf`` or ``.md``.
    """
    from llm_wiki.storage.object_store import RAW_PREFIX, get_object_store

    store = get_object_store()
    key: str | None
    if stored_key and store.exists(stored_key):
        key = stored_key
    else:
        key = next(
            (
                o.key
                for o in store.list_objects(RAW_PREFIX)
                if Path(o.key).stem == file_id
            ),
            None,
        )
    if key is None:
        raise FileNotFoundError(f"Raw object for {file_id!r} not found")

    ext = Path(key).suffix.lower()
    # Parsers read from a path; as_local_path downloads S3 keys to a tempfile.
    with store.as_local_path(key) as path:
        if ext == ".pdf":
            return parse_pdf(path)
        if ext == ".md":
            return parse_markdown_file(path).plain_text
    raise ValueError(f"Unsupported file extension: {ext!r}")


def _save_wiki_page(page: WikiPage) -> None:
    """Persist the wiki page to Postgres (source of truth + FTS in one write)."""
    from llm_wiki.storage import wiki_store

    wiki_store.save_page(page.slug, page.title, page.content)
    logger.debug("wiki_page_saved", slug=page.slug)


def _load_existing_pages(search_results: list[SearchHit]) -> list[WikiPage]:
    """Read wiki pages (from Postgres) for the slugs in *search_results*.

    Args:
        search_results: Ranked search results whose slugs to look up.

    Returns:
        WikiPage objects for results that have a corresponding stored page.
    """
    from llm_wiki.storage import wiki_store

    pages: list[WikiPage] = []
    for sr in search_results:
        content = wiki_store.get_page(sr.slug)
        if content is not None:
            pages.append(WikiPage(slug=sr.slug, title=sr.title, content=content))
    return pages


def _sum_cost(usage_log_path: Path, file_id: str) -> float:
    """Sum ``cost_usd`` from usage.log for all calls attributed to *file_id*.

    Args:
        usage_log_path: Path to the JSON-lines usage log.
        file_id: UUID to filter on.

    Returns:
        Total cost in USD, rounded to 6 decimal places.
    """
    if not usage_log_path.exists():
        return 0.0
    total = 0.0
    for line in usage_log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            if record.get("file_id") == file_id:
                total += float(record.get("cost_usd", 0.0))
        except (json.JSONDecodeError, TypeError):
            continue
    return round(total, 6)


def _run_linter_step(file_id: str) -> None:
    """Read all wiki pages and run the deterministic Linter.

    Writes the result to ``data/issues.md`` (AUTO_DETECTED section).  This
    is intentionally a *synchronous* helper so it can be called from the
    async pipeline without requiring an extra event loop context.

    Args:
        file_id: Used only for log correlation.

    Raises:
        Any exception is **not** caught here; the caller wraps this in
        try/except and decides whether to continue the pipeline.
    """
    from llm_wiki.quality.issues_writer import upsert_section
    from llm_wiki.quality.linter import run_linter
    from llm_wiki.quality.models import IssueSection
    from llm_wiki.storage.index import IndexStorage

    from llm_wiki.storage import wiki_store

    wiki_pages: dict[str, str] = dict(wiki_store.get_all_pages())

    # Derive root sections from index.md headings (level-2 headings = sections)
    index_storage = IndexStorage(settings.index_path)
    headings = index_storage.read_headings()
    # Root sections are those whose heading level is 2 (## Section Name)
    index_root_sections: set[str] = set()
    for h in headings:
        # Treat the first heading per section as a root slug placeholder
        # (slugified lower-case version of the section name)
        section_slug = h.section.lower().replace(" ", "-")
        index_root_sections.add(section_slug)

    from datetime import datetime, timezone

    current_year = datetime.now(timezone.utc).year
    issues = run_linter(
        wiki_pages=wiki_pages,
        index_root_sections=index_root_sections,
        current_year=current_year,
    )

    upsert_section(
        issues_path=settings.issues_path,
        section=IssueSection.AUTO_DETECTED,
        issues=issues,
    )

    counts = {}
    for issue in issues:
        counts[str(issue.kind)] = counts.get(str(issue.kind), 0) + 1
    logger.info(
        "linter_done",
        file_id=file_id,
        issues_found=len(issues),
        **counts,
    )
