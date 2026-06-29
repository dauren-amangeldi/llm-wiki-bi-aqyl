"""Integration tests for SHA-256 file deduplication in POST /files (LW-12.1)."""

import io
from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llm_wiki.storage.metadata import Base

# ``db_engine`` (Postgres, clean per test) comes from conftest.


@pytest.fixture
async def test_app(tmp_path: Path, db_engine):  # type: ignore[misc]
    from llm_wiki.main import app
    import llm_wiki.api.deps as deps_module

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
        bind=db_engine, expire_on_commit=False, autoflush=False
    )

    async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[deps_module.get_db] = _override_get_db
    yield app, tmp_path, raw_dir
    app.dependency_overrides.clear()


def _upload(content: bytes, filename: str = "doc.pdf") -> dict:  # type: ignore[type-arg]
    return {"file": (filename, io.BytesIO(content), "application/octet-stream")}


def _mock_settings(raw_dir: Path) -> MagicMock:
    m = MagicMock()
    m.raw_dir = raw_dir
    m.max_file_size_mb = 50
    m.allowed_extensions = frozenset({".pdf", ".md"})
    return m


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_first_upload_returns_202(test_app: tuple) -> None:  # type: ignore[type-arg]
    """A fresh upload returns 202 with status='queued'."""
    app, _, raw_dir = test_app
    mock_settings = _mock_settings(raw_dir)

    with (
        patch("llm_wiki.api.routes.process_file_task") as mock_task,
        patch("llm_wiki.api.routes.settings", mock_settings),
    ):
        mock_task.delay.return_value = MagicMock(id="task-1")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post("/api/v1/files", files=_upload(b"%PDF-1.4 unique content"))

    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "queued"
    assert body["task_id"] == "task-1"
    assert body["duplicate_of"] is None


async def test_duplicate_upload_returns_200(test_app: tuple) -> None:  # type: ignore[type-arg]
    """Uploading the same bytes twice returns 200 with status='duplicate'."""
    app, _, raw_dir = test_app
    mock_settings = _mock_settings(raw_dir)
    content = b"%PDF-1.4 identical content"

    with (
        patch("llm_wiki.api.routes.process_file_task") as mock_task,
        patch("llm_wiki.api.routes.settings", mock_settings),
    ):
        mock_task.delay.return_value = MagicMock(id="task-A")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            first = await c.post("/api/v1/files", files=_upload(content, "a.pdf"))

    original_file_id = first.json()["file_id"]

    with (
        patch("llm_wiki.api.routes.process_file_task") as mock_task2,
        patch("llm_wiki.api.routes.settings", mock_settings),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            second = await c.post("/api/v1/files", files=_upload(content, "b.pdf"))

    assert second.status_code == 200
    body = second.json()
    assert body["status"] == "duplicate"
    assert body["file_id"] == original_file_id
    assert body["duplicate_of"] == original_file_id
    assert body["task_id"] is None
    # Celery task must NOT have been enqueued for the duplicate
    mock_task2.delay.assert_not_called()


async def test_duplicate_does_not_enqueue_celery(test_app: tuple) -> None:  # type: ignore[type-arg]
    """Celery pipeline is skipped for duplicate content."""
    app, _, raw_dir = test_app
    mock_settings = _mock_settings(raw_dir)
    content = b"# Duplicate markdown content"

    with (
        patch("llm_wiki.api.routes.process_file_task") as task1,
        patch("llm_wiki.api.routes.settings", mock_settings),
    ):
        task1.delay.return_value = MagicMock(id="first-task")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            await c.post("/api/v1/files", files=_upload(content, "first.md"))

    with (
        patch("llm_wiki.api.routes.process_file_task") as task2,
        patch("llm_wiki.api.routes.settings", mock_settings),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            await c.post("/api/v1/files", files=_upload(content, "second.md"))

    task2.delay.assert_not_called()


async def test_upload_after_failed_allows_retry(tmp_path: Path, db_engine) -> None:  # type: ignore[misc,type-arg]
    """Uploading identical content after a FAILED ingestion is allowed."""
    from llm_wiki.main import app
    import llm_wiki.api.deps as deps_module

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    mock_settings = _mock_settings(raw_dir)

    session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
        bind=db_engine, expire_on_commit=False, autoflush=False
    )

    async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[deps_module.get_db] = _override_get_db

    content = b"%PDF-1.4 will fail"

    try:
        # First upload
        with (
            patch("llm_wiki.api.routes.process_file_task") as task1,
            patch("llm_wiki.api.routes.settings", mock_settings),
        ):
            task1.delay.return_value = MagicMock(id="task-fail")
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                first_resp = await c.post(
                    "/api/v1/files", files=_upload(content, "fail.pdf")
                )

        first_id = first_resp.json()["file_id"]

        # Simulate FAILED status
        from sqlalchemy import update as sa_update
        async with session_factory() as session:
            await session.execute(
                sa_update(Base.metadata.tables["files"])
                .where(Base.metadata.tables["files"].c.file_id == first_id)
                .values(status="FAILED")
            )
            await session.commit()

        # Second upload of same content — should be allowed (new task queued)
        with (
            patch("llm_wiki.api.routes.process_file_task") as task2,
            patch("llm_wiki.api.routes.settings", mock_settings),
        ):
            task2.delay.return_value = MagicMock(id="task-retry")
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                second_resp = await c.post(
                    "/api/v1/files", files=_upload(content, "fail_retry.pdf")
                )

        assert second_resp.status_code == 202
        assert second_resp.json()["status"] == "queued"
        task2.delay.assert_called_once()

    finally:
        app.dependency_overrides.clear()


async def test_empty_file_returns_400(test_app: tuple) -> None:  # type: ignore[type-arg]
    """An empty upload body is rejected with 400."""
    app, _, raw_dir = test_app
    mock_settings = _mock_settings(raw_dir)

    with patch("llm_wiki.api.routes.settings", mock_settings):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post("/api/v1/files", files=_upload(b"", "empty.pdf"))

    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()
