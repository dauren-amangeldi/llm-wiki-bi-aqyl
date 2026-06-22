"""Tests for notebook-only ingestion (LW-N16) and library attach (Task 2)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from llm_wiki.orchestrator.pipeline import (
    attach_notebook_existing_file,
    index_notebook_source,
)
from llm_wiki.storage.metadata import Base, create_file_record, create_notebook


@pytest.fixture
async def db_session(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/nb.db")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)
    async with factory() as session:
        yield session
    await engine.dispose()


def test_index_notebook_source_uses_nb_slug_prefix() -> None:
    """Notebook sources use nb-{file_id} slug — never wiki slugs."""
    chunk_store = MagicMock()
    chunk_store.upsert_page = MagicMock()

    index_notebook_source(chunk_store, "abc-123", "Report", "Hello world content")

    chunk_store.upsert_page.assert_called_once()
    kwargs = chunk_store.upsert_page.call_args.kwargs
    assert kwargs["slug"] == "nb-abc-123"
    assert kwargs["file_id"] == "abc-123"
    assert kwargs["title"] == "Report"


def test_index_notebook_source_truncates_long_text() -> None:
    """Very long uploads are truncated before embedding."""
    chunk_store = MagicMock()
    long_text = "x" * 200_000
    index_notebook_source(chunk_store, "f1", "Big", long_text)
    content = chunk_store.upsert_page.call_args.kwargs["content"]
    assert len(content) <= 120_000


@pytest.mark.asyncio
async def test_attach_existing_file_backfills_when_no_chunks(
    db_session: AsyncSession,
    tmp_path,
) -> None:
    """Attach from library indexes raw text when file_id has zero chunks."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "global-file.md").write_text(
        "# Case\n\nUnique notebook content xyzzy.",
        encoding="utf-8",
    )

    nb = await create_notebook(db_session, "user-a", "Research")
    await create_file_record(db_session, "global-file", "global-file.md")

    chunk_store = MagicMock()
    chunk_store.count_by_file_id.return_value = 0

    with patch("llm_wiki.orchestrator.pipeline.settings") as mock_settings, patch(
        "llm_wiki.orchestrator.pipeline.index_notebook_source"
    ) as mock_index:
        mock_settings.raw_dir = raw_dir
        await attach_notebook_existing_file(
            db_session, nb.id, "user-a", "global-file", chunk_store
        )
        mock_index.assert_called_once()
        assert mock_index.call_args.args[1] == "global-file"


@pytest.mark.asyncio
async def test_attach_existing_file_always_indexes_from_raw(
    db_session: AsyncSession,
    tmp_path,
) -> None:
    """Library attach always embeds raw text into nb-{file_id}, even if wiki chunks exist."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "file-1.md").write_text(
        "# Case\n\nNotebook raw content xyzzy.",
        encoding="utf-8",
    )

    nb = await create_notebook(db_session, "user-a", "Research")
    await create_file_record(db_session, "file-1", "doc.md")

    chunk_store = MagicMock()
    chunk_store.count_by_file_id.return_value = 3

    with patch("llm_wiki.orchestrator.pipeline.settings") as mock_settings, patch(
        "llm_wiki.orchestrator.pipeline.index_notebook_source"
    ) as mock_index:
        mock_settings.raw_dir = raw_dir
        await attach_notebook_existing_file(
            db_session, nb.id, "user-a", "file-1", chunk_store
        )
        mock_index.assert_called_once()
        assert mock_index.call_args.args[1] == "file-1"

    chunk_store.count_by_file_id.assert_not_called()
