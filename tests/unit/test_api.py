"""Unit tests for the API layer — POST /files endpoint (LW-5)."""

import io
from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from llm_wiki.storage.metadata import Base


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def db_engine(tmp_path: Path):  # type: ignore[misc]
    """In-memory SQLite engine with schema created."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/test.db")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def test_app(tmp_path: Path, db_engine):  # type: ignore[misc]
    """FastAPI app wired to a temp DB and temp data directories."""
    from llm_wiki.main import app
    import llm_wiki.api.deps as deps_module

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    # Override the DB session dependency
    session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
        bind=db_engine, expire_on_commit=False, autoflush=False
    )

    async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[deps_module.get_db] = _override_get_db

    yield app, tmp_path, raw_dir

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_upload(content: bytes, filename: str) -> dict:  # type: ignore[type-arg]
    """Build the files dict for httpx multipart upload."""
    return {"file": (filename, io.BytesIO(content), "application/octet-stream")}


def _mock_settings(raw_dir: Path) -> MagicMock:
    """Return a mock settings with raw_dir pointing at tmp_path."""
    m = MagicMock()
    m.raw_dir = raw_dir
    m.max_file_size_mb = 50
    m.allowed_extensions = frozenset({".pdf", ".md"})
    m.service_name = "llm-wiki-test"
    return m


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_health_check(test_app: tuple) -> None:  # type: ignore[type-arg]
    """GET /health returns 200."""
    app, *_ = test_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_upload_pdf_returns_202(test_app: tuple, tmp_path: Path) -> None:  # type: ignore[type-arg]
    """POST /files with a valid PDF returns 202 with file_id and task_id."""
    app, data_tmp, raw_dir = test_app
    fake_pdf = b"%PDF-1.4 fake content that is long enough"

    mock_settings = _mock_settings(raw_dir)

    with (
        patch("llm_wiki.api.routes.process_file_task") as mock_task,
        patch("llm_wiki.api.routes.settings", mock_settings),
    ):
        mock_task.delay.return_value = MagicMock(id="celery-task-123")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/files", files=_make_upload(fake_pdf, "report.pdf"))

    assert resp.status_code == 202
    body = resp.json()
    assert "file_id" in body
    assert body["task_id"] == "celery-task-123"
    assert body["status"] == "queued"


async def test_upload_md_returns_202(test_app: tuple, tmp_path: Path) -> None:  # type: ignore[type-arg]
    """POST /files with a valid Markdown file returns 202."""
    app, data_tmp, raw_dir = test_app
    md_content = b"# Hello\n\nSome content here.\n"
    mock_settings = _mock_settings(raw_dir)

    with (
        patch("llm_wiki.api.routes.process_file_task") as mock_task,
        patch("llm_wiki.api.routes.settings", mock_settings),
    ):
        mock_task.delay.return_value = MagicMock(id="celery-md-456")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/files", files=_make_upload(md_content, "notes.md"))

    assert resp.status_code == 202
    assert resp.json()["status"] == "queued"


async def test_upload_unsupported_type_returns_400(test_app: tuple) -> None:  # type: ignore[type-arg]
    """POST /files with a .txt file returns 400."""
    app, _, raw_dir = test_app
    mock_settings = _mock_settings(raw_dir)

    with (
        patch("llm_wiki.api.routes.process_file_task"),
        patch("llm_wiki.api.routes.settings", mock_settings),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/files", files=_make_upload(b"plain text", "readme.txt")
            )
    assert resp.status_code == 400
    assert "Unsupported file type" in resp.json()["detail"]


async def test_upload_too_large_returns_413(test_app: tuple) -> None:  # type: ignore[type-arg]
    """POST /files with a file over 50 MB returns 413."""
    app, _, raw_dir = test_app
    mock_settings = _mock_settings(raw_dir)
    # Content just over the 50 MB limit
    oversized = b"x" * (51 * 1024 * 1024)

    with (
        patch("llm_wiki.api.routes.process_file_task"),
        patch("llm_wiki.api.routes.settings", mock_settings),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/files", files=_make_upload(oversized, "big.pdf")
            )
    assert resp.status_code == 413


async def test_upload_saves_file_to_raw(test_app: tuple) -> None:  # type: ignore[type-arg]
    """Uploaded file must be saved to raw_dir with file_id as stem."""
    app, data_tmp, raw_dir = test_app
    content = b"%PDF-1.4 test"
    mock_settings = _mock_settings(raw_dir)

    with (
        patch("llm_wiki.api.routes.process_file_task") as mock_task,
        patch("llm_wiki.api.routes.settings", mock_settings),
    ):
        mock_task.delay.return_value = MagicMock(id="t1")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/files", files=_make_upload(content, "doc.pdf"))

    file_id = resp.json()["file_id"]
    saved = raw_dir / f"{file_id}.pdf"
    assert saved.exists(), f"Expected {saved} to exist"
    assert saved.read_bytes() == content


async def test_upload_creates_db_record(test_app: tuple, db_engine) -> None:  # type: ignore[type-arg, misc]
    """Uploading a file creates a FileRecord in the database."""
    app, data_tmp, raw_dir = test_app
    content = b"# Markdown content"
    mock_settings = _mock_settings(raw_dir)

    with (
        patch("llm_wiki.api.routes.process_file_task") as mock_task,
        patch("llm_wiki.api.routes.settings", mock_settings),
    ):
        mock_task.delay.return_value = MagicMock(id="t2")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/files", files=_make_upload(content, "page.md"))

    file_id = resp.json()["file_id"]

    from llm_wiki.storage.metadata import get_file_record

    session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
        bind=db_engine, expire_on_commit=False
    )
    async with session_factory() as session:
        record = await get_file_record(session, file_id)

    assert record is not None
    assert record.original_name == "page.md"
    assert record.status == "RECEIVED"
