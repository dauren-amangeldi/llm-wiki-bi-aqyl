"""Integration tests for the ingestion pipeline (LW-9).

Tests exercise ``process_file`` end-to-end using:
 - an in-memory SQLite DB (no Docker required)
 - patched ``LLMClient`` so no real LLM calls are made
 - patched ``settings`` pointing at ``tmp_path``
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from llm_wiki.agents.search import SearchResult
from llm_wiki.agents.writer import WikiPage
from llm_wiki.storage.metadata import Base, FileRecord, create_file_record, get_file_record


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def db_engine(tmp_path: Path):  # type: ignore[misc]
    """Create an in-memory SQLite engine with the full schema."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/meta.db")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def db_session(db_engine) -> AsyncSession:  # type: ignore[misc]
    """Yield a single async session for the test DB."""
    factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
        bind=db_engine, expire_on_commit=False, autoflush=False
    )
    async with factory() as session:
        yield session


@pytest.fixture
def pipeline_dirs(tmp_path: Path) -> dict[str, Path]:
    """Create the full data-dir layout under tmp_path."""
    dirs = {
        "data_dir": tmp_path,
        "raw": tmp_path / "raw",
        "wiki": tmp_path / "wiki",
        "chroma": tmp_path / "chroma",
    }
    for p in dirs.values():
        p.mkdir(exist_ok=True)
    (tmp_path / "index.md").write_text("# Wiki Index\n")
    (tmp_path / "log.md").write_text("# Ingestion Log\n")
    (tmp_path / "usage.log").write_text("")
    return dirs


@pytest.fixture
def fake_md_file(pipeline_dirs: dict[str, Path]) -> tuple[str, Path]:
    """Write a fake .md file in raw/ and return (file_id, path)."""
    file_id = "test-file-id-001"
    path = pipeline_dirs["raw"] / f"{file_id}.md"
    path.write_text("# Transformer Architecture\n\nDeep learning model intro.\n")
    return file_id, path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings_patch(pipeline_dirs: dict[str, Path], db_engine) -> MagicMock:  # type: ignore[misc]
    """Build a mock settings object pointing at tmp_path."""
    mock_settings = MagicMock()
    mock_settings.raw_dir = pipeline_dirs["raw"]
    mock_settings.wiki_dir = pipeline_dirs["wiki"]
    mock_settings.chroma_dir = pipeline_dirs["chroma"]
    mock_settings.index_path = pipeline_dirs["data_dir"] / "index.md"
    mock_settings.log_path = pipeline_dirs["data_dir"] / "log.md"
    mock_settings.usage_log_path = pipeline_dirs["data_dir"] / "usage.log"
    mock_settings.llm_provider = "ollama"
    mock_settings.ollama_base_url = "http://localhost:11434"
    mock_settings.ollama_model = "test"
    mock_settings.openai_api_key = ""
    mock_settings.openai_model = "test"
    mock_settings.anthropic_api_key = ""
    mock_settings.anthropic_model = "test"
    mock_settings.price_table = {}
    return mock_settings


def _llm_create_response(slug: str = "transformers", title: str = "Transformers") -> tuple[str, MagicMock]:
    """JSON response for a SearchAgent or WriterAgent create_page call."""
    return (
        json.dumps({"slug": slug, "title": title, "content": f"# {title}\n\nContent."}),
        MagicMock(),
    )


def _llm_search_empty_response() -> tuple[str, MagicMock]:
    """JSON response: no relevant pages found."""
    return (json.dumps([]), MagicMock())


def _llm_search_response(slug: str = "transformers") -> tuple[str, MagicMock]:
    """JSON response: one relevant page with high score."""
    return (
        json.dumps([{"slug": slug, "title": "Transformers", "relevance_score": 0.9}]),
        MagicMock(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_upload_md_creates_wiki_page(
    db_engine, db_session: AsyncSession, pipeline_dirs: dict[str, Path], fake_md_file: tuple[str, Path],
) -> None:
    """process_file with an .md file creates a wiki page and logs the event."""
    file_id, _path = fake_md_file

    await create_file_record(db_session, file_id, "architecture.md")

    mock_settings = _make_settings_patch(pipeline_dirs, db_engine)

    # LLM returns: search finds no pages → create new
    call_count = 0

    async def _fake_complete(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1
        if kwargs.get("agent_type") == "search":
            return _llm_search_empty_response()
        return _llm_create_response()

    mock_llm = MagicMock()
    mock_llm.load_prompt.return_value = "prompt"
    mock_llm.complete = AsyncMock(side_effect=_fake_complete)

    factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
        bind=db_engine, expire_on_commit=False, autoflush=False
    )

    with (
        patch("llm_wiki.orchestrator.pipeline.settings", mock_settings),
        patch("llm_wiki.orchestrator.pipeline.LLMClient", return_value=mock_llm),
        patch("llm_wiki.api.deps._engine", db_engine),
    ):
        from llm_wiki.orchestrator.pipeline import process_file

        await process_file(file_id)

    # Wiki page exists
    wiki_page = pipeline_dirs["wiki"] / "transformers.md"
    assert wiki_page.exists(), f"Expected {wiki_page} to be created"

    # index.md updated
    index_text = (pipeline_dirs["data_dir"] / "index.md").read_text()
    assert "transformers" in index_text

    # log.md updated
    log_text = (pipeline_dirs["data_dir"] / "log.md").read_text()
    assert file_id in log_text

    # DB status = DONE
    async with factory() as s:
        record = await get_file_record(s, file_id)
    assert record is not None
    assert record.status == "DONE"


async def test_pipeline_idempotent_on_rerun(
    db_engine, db_session: AsyncSession, pipeline_dirs: dict[str, Path], fake_md_file: tuple[str, Path],
) -> None:
    """Running process_file twice must not duplicate wiki pages or log entries."""
    file_id, _path = fake_md_file

    await create_file_record(db_session, file_id, "architecture.md")

    mock_settings = _make_settings_patch(pipeline_dirs, db_engine)
    mock_llm = MagicMock()
    mock_llm.load_prompt.return_value = "prompt"
    mock_llm.complete = AsyncMock(
        side_effect=lambda **kw: _llm_search_empty_response()
        if kw.get("agent_type") == "search"
        else _llm_create_response()
    )

    with (
        patch("llm_wiki.orchestrator.pipeline.settings", mock_settings),
        patch("llm_wiki.orchestrator.pipeline.LLMClient", return_value=mock_llm),
        patch("llm_wiki.api.deps._engine", db_engine),
    ):
        from llm_wiki.orchestrator.pipeline import process_file

        await process_file(file_id)
        await process_file(file_id)

    # Only one wiki page
    wiki_files = list(pipeline_dirs["wiki"].glob("*.md"))
    assert len(wiki_files) == 1, f"Expected 1 wiki page, found {wiki_files}"

    # log.md contains file_id exactly once
    log_text = (pipeline_dirs["data_dir"] / "log.md").read_text()
    assert log_text.count(file_id) == 1


async def test_pipeline_failed_state_on_llm_error(
    db_engine, db_session: AsyncSession, pipeline_dirs: dict[str, Path], fake_md_file: tuple[str, Path],
) -> None:
    """When the LLM raises on the search step, the file ends in FAILED state."""
    file_id, _path = fake_md_file

    await create_file_record(db_session, file_id, "architecture.md")

    mock_settings = _make_settings_patch(pipeline_dirs, db_engine)
    mock_llm = MagicMock()
    mock_llm.load_prompt.return_value = "prompt"
    mock_llm.complete = AsyncMock(side_effect=RuntimeError("LLM exploded"))

    factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
        bind=db_engine, expire_on_commit=False, autoflush=False
    )

    with (
        patch("llm_wiki.orchestrator.pipeline.settings", mock_settings),
        patch("llm_wiki.orchestrator.pipeline.LLMClient", return_value=mock_llm),
        patch("llm_wiki.api.deps._engine", db_engine),
        pytest.raises(RuntimeError, match="LLM exploded"),
    ):
        from llm_wiki.orchestrator.pipeline import process_file

        await process_file(file_id)

    async with factory() as s:
        record = await get_file_record(s, file_id)

    assert record is not None
    assert record.status == "FAILED"
