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
from llm_wiki.storage.filesystem import atomic_write
from llm_wiki.storage.index import IndexStorage
from llm_wiki.storage.wiki_fts import upsert_wiki_fts
from llm_wiki.utils.backlinks import extract_outgoing_links
from llm_wiki.storage.log import append_log_entry
from llm_wiki.utils.ids import new_file_id
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

        # One LLMClient for the whole pipeline run; closed in finally so its
        # httpx connections are released before the event loop shuts down.
        llm = LLMClient()
        try:
            # ----------------------------------------------------------------
            # STORED — parse raw file to plain text
            # ----------------------------------------------------------------
            raw_path = _find_raw_file(settings.raw_dir, file_id)
            file_text = _parse_raw_file(raw_path)

            if "STORED" not in completed:
                await _transition(session, file_id, "STORED")

            # ----------------------------------------------------------------
            # SEARCHED — Search Agent v2 (embedding pre-filter + LLM re-rank)
            # ----------------------------------------------------------------
            embedding_store = EmbeddingStore(
                chroma_path=settings.chroma_dir, llm_client=llm
            )
            chunk_store = ChunkStore(
                chroma_path=settings.chroma_dir, llm_client=llm
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
                    await _index_wiki_page_fts(session, page)
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
                        wiki_dir=settings.wiki_dir,
                        source_slug=page.slug,
                        new_content=page.content,
                        previous_outgoing=(),
                        file_id=file_id,
                    )
                else:
                    # Scenario B — update existing pages (up to 5)
                    existing = _load_existing_pages(settings.wiki_dir, search_results[:5])

                    if existing:
                        # Capture outgoing links BEFORE the Writer Agent rewrites pages
                        previous_outgoing_by_slug: dict[str, list[str]] = {
                            p.slug: extract_outgoing_links(p.content) for p in existing
                        }
                        pages_out = await writer.update_pages(file_text, existing, file_id)
                        for p in pages_out:
                            _save_wiki_page(settings.wiki_dir, p)
                            await _index_wiki_page_fts(session, p)
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
                                wiki_dir=settings.wiki_dir,
                                source_slug=p.slug,
                                new_content=p.content,
                                previous_outgoing=previous_outgoing_by_slug.get(p.slug, []),
                                file_id=file_id,
                            )
                    else:
                        # Search found headings but files are absent — create new
                        page = await writer.create_page(file_text, file_id)
                        _save_wiki_page(settings.wiki_dir, page)
                        await _index_wiki_page_fts(session, page)
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
                            wiki_dir=settings.wiki_dir,
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
# Notebook-scoped ingestion (LW-N16) — never touches wiki / Search Agent
# ---------------------------------------------------------------------------


def index_notebook_source(
    chunk_store: ChunkStore,
    file_id: str,
    title: str,
    text: str,
) -> None:
    """Chunk and embed raw source text for notebook-scoped retrieval.

    Uses slug ``nb-{file_id}`` so notebook sources never collide with wiki
    page slugs.  Does not write to ``data/wiki/`` or ``index.md``.
    """
    truncated = text[: settings.notebook_max_source_chars]
    slug = f"nb-{file_id}"
    chunk_store.upsert_page(slug=slug, title=title, content=truncated, file_id=file_id)
    logger.info(
        "notebook_source_indexed",
        file_id=file_id,
        slug=slug,
        chars=len(truncated),
    )


async def ingest_notebook_file(
    session: AsyncSession,
    notebook_id: str,
    owner_id: str,
    filename: str,
    content: bytes,
    llm: LLMClient,
    chunk_store: ChunkStore,
) -> str:
    """Save, embed (if needed), and attach a file to a notebook.

    Skips re-embedding when the same ``file_id`` already has chunks — only
    adds a ``notebook_files`` link so one file can live in many notebooks.

    Args:
        session: Active async SQLAlchemy session.
        notebook_id: Target notebook UUID.
        owner_id: Must own the notebook.
        filename: Original upload name.
        content: Raw file bytes.
        llm: LLM client for embeddings.
        chunk_store: Chroma chunk store.

    Returns:
        The ``file_id`` attached to the notebook.

    Raises:
        LookupError: Notebook not found for owner.
        ValueError: Unsupported extension or empty file.
    """
    import io

    from llm_wiki.storage.metadata import attach_file, create_file_record, get_by_sha256, get_notebook
    from llm_wiki.utils.hashing import sha256_stream

    notebook = await get_notebook(session, notebook_id, owner_id)
    if notebook is None:
        raise LookupError(f"Notebook {notebook_id!r} not found for owner {owner_id!r}")

    ext = Path(filename).suffix.lower()
    if ext not in settings.allowed_extensions:
        raise ValueError(
            f"Unsupported file type {ext!r}. Allowed: {sorted(settings.allowed_extensions)}"
        )
    if len(content) == 0:
        raise ValueError("Empty file is not allowed.")

    sha = sha256_stream(io.BytesIO(content))
    existing = await get_by_sha256(session, sha)

    if existing is not None:
        file_id = existing.file_id
        if chunk_store.count_by_file_id(file_id) == 0:
            raw_path = _find_raw_file(settings.raw_dir, file_id)
            text = _parse_raw_file(raw_path)
            index_notebook_source(chunk_store, file_id, existing.original_name, text)
    else:
        file_id = new_file_id()
        dest = settings.raw_dir / f"{file_id}{ext}"
        dest.write_bytes(content)
        await create_file_record(session, file_id, filename, content_sha256=sha)
        await update_file_status(session, file_id, "NOTEBOOK")
        text = _parse_raw_file(dest)
        index_notebook_source(chunk_store, file_id, filename, text)

    await attach_file(session, notebook_id, file_id, owner_id)
    return file_id


async def attach_notebook_existing_file(
    session: AsyncSession,
    notebook_id: str,
    owner_id: str,
    file_id: str,
    chunk_store: ChunkStore,
) -> None:
    """Link an existing ingested file to a notebook.

    Always embeds raw text from ``data/raw/`` under slug ``nb-{file_id}`` so
    notebook Q&A uses source content, even when global wiki chunks already
    exist for the same ``file_id``.  Does not write to ``data/wiki/`` or run
    the Search Agent.
    """
    from llm_wiki.storage.metadata import attach_file, get_notebook

    notebook = await get_notebook(session, notebook_id, owner_id)
    if notebook is None:
        raise LookupError(f"Notebook {notebook_id!r} not found for owner {owner_id!r}")
    record = await get_file_record(session, file_id)
    if record is None:
        raise LookupError(f"File {file_id!r} not found")
    try:
        raw_path = _find_raw_file(settings.raw_dir, file_id)
    except FileNotFoundError as exc:
        raise LookupError(f"Raw file for {file_id!r} not found") from exc
    text = _parse_raw_file(raw_path)
    index_notebook_source(chunk_store, file_id, record.original_name, text)
    await attach_file(session, notebook_id, file_id, owner_id)


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


async def _index_wiki_page_fts(session: AsyncSession, page: WikiPage) -> None:
    """Update the FTS5 index after a wiki page is saved (LW-N4)."""
    try:
        await upsert_wiki_fts(session, page.slug, page.title, page.content)
    except Exception as exc:  # noqa: BLE001
        logger.error("wiki_fts_index_failed", slug=page.slug, error=str(exc))


def _load_existing_pages(
    wiki_dir: Path, search_results: list[SearchHit]
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

    wiki_dir = settings.wiki_dir
    wiki_pages: dict[str, str] = {}
    if wiki_dir.exists():
        for md_file in wiki_dir.glob("*.md"):
            slug = md_file.stem
            wiki_pages[slug] = md_file.read_text(encoding="utf-8")

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
