"""Ingestion pipeline state machine.

State transitions:
    RECEIVED → STORED → SEARCHED → WRITTEN → LOGGED → DONE
                                ↘ FAILED (with step + error)

Each step is idempotent: safe to re-run with the same file_id.
"""

import json
from enum import StrEnum
from pathlib import Path

import structlog
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llm_wiki.agents.search import SearchAgent, SearchResult
from llm_wiki.agents.writer import WikiPage, WriterAgent
from llm_wiki.config import settings
from llm_wiki.llm.client import LLMClient
from llm_wiki.parsers.markdown import parse_markdown_file
from llm_wiki.parsers.pdf import parse_pdf
from llm_wiki.storage.filesystem import atomic_write
from llm_wiki.storage.index import IndexStorage
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

        try:
            # ----------------------------------------------------------------
            # STORED — parse raw file to plain text
            # ----------------------------------------------------------------
            raw_path = _find_raw_file(settings.raw_dir, file_id)
            file_text = _parse_raw_file(raw_path)

            if "STORED" not in completed:
                await _transition(session, file_id, "STORED")

            # ----------------------------------------------------------------
            # SEARCHED — Search Agent finds relevant pages
            # ----------------------------------------------------------------
            llm = LLMClient()
            index_storage = IndexStorage(settings.index_path)
            headings = index_storage.read_headings()
            heading_texts = [h.text for h in headings]

            search_agent = SearchAgent(llm)
            search_results: list[SearchResult] = await search_agent.run(
                file_text, heading_texts, file_id=file_id
            )

            if "SEARCHED" not in completed:
                await _transition(session, file_id, "SEARCHED")

            # ----------------------------------------------------------------
            # WRITTEN — Writer Agent creates / updates pages
            # ----------------------------------------------------------------
            if "WRITTEN" not in completed:
                writer = WriterAgent(llm)
                created_pages: list[str] = []
                updated_pages: list[str] = []

                if not search_results:
                    # Scenario A — brand-new topic
                    page = await writer.create_page(file_text, file_id)
                    _save_wiki_page(settings.wiki_dir, page)
                    index_storage.add_page(page.slug, "General")
                    created_pages.append(page.slug)
                else:
                    # Scenario B — update existing pages (up to 5)
                    existing = _load_existing_pages(settings.wiki_dir, search_results[:5])

                    if existing:
                        pages_out = await writer.update_pages(file_text, existing, file_id)
                        for p in pages_out:
                            _save_wiki_page(settings.wiki_dir, p)
                            updated_pages.append(p.slug)
                    else:
                        # Search found headings but files are absent — create new
                        page = await writer.create_page(file_text, file_id)
                        _save_wiki_page(settings.wiki_dir, page)
                        index_storage.add_page(page.slug, "General")
                        created_pages.append(page.slug)

                # Persist page lists to DB
                from sqlalchemy import update as sa_update

                await session.execute(
                    sa_update(FileRecord)
                    .where(FileRecord.file_id == file_id)
                    .values(created_pages=created_pages, updated_pages=updated_pages)
                )
                await session.commit()
                await _transition(session, file_id, "WRITTEN")

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


def _find_raw_file(raw_dir: Path, file_id: str) -> Path:
    """Return the raw file path for *file_id*, scanning *raw_dir* by stem.

    Args:
        raw_dir: Directory containing uploaded source files.
        file_id: UUID used as the file stem.

    Returns:
        Path to the matching raw file.

    Raises:
        FileNotFoundError: If no file with stem *file_id* exists.
    """
    for candidate in raw_dir.iterdir():
        if candidate.stem == file_id:
            return candidate
    raise FileNotFoundError(f"Raw file for {file_id!r} not found in {raw_dir}")


def _parse_raw_file(path: Path) -> str:
    """Extract plain text from a raw file depending on its extension.

    Args:
        path: Path to a PDF or Markdown file.

    Returns:
        Extracted plain text.

    Raises:
        ValueError: If the file extension is not ``.pdf`` or ``.md``.
    """
    ext = path.suffix.lower()
    if ext == ".pdf":
        return parse_pdf(path)
    if ext == ".md":
        return parse_markdown_file(path).plain_text
    raise ValueError(f"Unsupported file extension: {ext!r}")


def _save_wiki_page(wiki_dir: Path, page: WikiPage) -> None:
    """Atomically write *page.content* to wiki_dir/{slug}.md.

    Args:
        wiki_dir: Directory for generated wiki pages.
        page: The WikiPage to persist.
    """
    dest = wiki_dir / f"{page.slug}.md"
    atomic_write(dest, page.content)
    logger.debug("wiki_page_saved", slug=page.slug, path=str(dest))


def _load_existing_pages(
    wiki_dir: Path, search_results: list[SearchResult]
) -> list[WikiPage]:
    """Read wiki files for *search_results* that actually exist on disk.

    Args:
        wiki_dir: Directory containing wiki markdown files.
        search_results: Ranked search results whose slugs to look up.

    Returns:
        WikiPage objects for results that have a corresponding .md file.
    """
    pages: list[WikiPage] = []
    for sr in search_results:
        wiki_path = wiki_dir / f"{sr.slug}.md"
        if wiki_path.exists():
            content = wiki_path.read_text(encoding="utf-8")
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
