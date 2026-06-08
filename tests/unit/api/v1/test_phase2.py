"""Phase 2 integration tests — upload, status polling, source deletion.

Covers:
  POST /api/v1/uploads          — single file (DropZone contract)
  POST /api/v1/materials/upload — batch with skipped-duplicate tracking
  GET  /api/v1/documents/{id}/status — polling after upload
  DELETE /api/v1/documents/{id}/sources/{source_id}

Test strategy mirrors test_phase1.py:
  - In-memory SQLite, ``get_db`` dependency overridden.
  - ``settings.data_dir`` redirected to ``tmp_path``.
  - Celery ``process_file_task.delay`` is always mocked so no worker is needed.
"""

from __future__ import annotations

import io
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from llm_wiki.api.deps import get_db
from llm_wiki.config import settings
from llm_wiki.main import app
from llm_wiki.storage.metadata import Base, FileRecord


# ---------------------------------------------------------------------------
# Fixtures  (mirror test_phase1.py)
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def db_engine():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(db_engine) -> AsyncGenerator[AsyncSession, None]:
    factory = async_sessionmaker(bind=db_engine, expire_on_commit=False, autoflush=False)
    async with factory() as session:
        yield session


@pytest_asyncio.fixture
async def client(
    tmp_path: Path,
    db_engine,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncGenerator[AsyncClient, None]:
    object.__setattr__(settings, "data_dir", tmp_path)
    (tmp_path / "wiki").mkdir(exist_ok=True)
    (tmp_path / "raw").mkdir(exist_ok=True)

    factory = async_sessionmaker(bind=db_engine, expire_on_commit=False, autoflush=False)

    async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with factory() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db

    async with AsyncClient(
        transport=ASGITransport(app=app),  # type: ignore[arg-type]
        base_url="http://test",
    ) as ac:
        yield ac

    app.dependency_overrides.clear()


def _file(content: bytes = b"%PDF-1.4 test", name: str = "doc.pdf") -> dict:
    """Build an httpx multipart files dict for a single file upload."""
    return {"file": (name, io.BytesIO(content), "application/octet-stream")}


def _files_batch(*pairs: tuple[str, bytes]) -> list[tuple[str, tuple]]:
    """Build an httpx multipart files list for a batch upload (multi ``file`` field)."""
    return [("files", (name, io.BytesIO(data), "application/octet-stream")) for name, data in pairs]


# ---------------------------------------------------------------------------
# POST /api/v1/uploads  — single file
# ---------------------------------------------------------------------------


class TestUploadSingle:
    async def test_happy_pdf(self, client: AsyncClient) -> None:
        with patch("llm_wiki.api.v1.uploads.process_file_task") as mock_task:
            mock_task.delay.return_value = MagicMock(id="task-1")
            r = await client.post("/api/v1/uploads", files=_file())

        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "queued"
        assert data["content_type"] == "pdf"
        assert data["title"] == "doc"
        assert data["path"].endswith("/raw")
        assert "document_id" in data
        mock_task.delay.assert_called_once()

    async def test_happy_markdown(self, client: AsyncClient) -> None:
        with patch("llm_wiki.api.v1.uploads.process_file_task") as mock_task:
            mock_task.delay.return_value = MagicMock(id="task-2")
            r = await client.post("/api/v1/uploads", files=_file(b"# Hello", "note.md"))

        assert r.status_code == 200
        assert r.json()["content_type"] == "markdown"

    async def test_duplicate_returns_existing_document_id(self, client: AsyncClient) -> None:
        content = b"%PDF-1.4 duplicate bytes"
        first_id: str | None = None

        with patch("llm_wiki.api.v1.uploads.process_file_task") as mock_task:
            mock_task.delay.return_value = MagicMock(id="t1")
            r1 = await client.post("/api/v1/uploads", files=_file(content, "a.pdf"))
        first_id = r1.json()["document_id"]

        with patch("llm_wiki.api.v1.uploads.process_file_task") as mock_task2:
            r2 = await client.post("/api/v1/uploads", files=_file(content, "b.pdf"))

        assert r2.status_code == 200
        data = r2.json()
        assert data["status"] == "duplicate"
        assert data["document_id"] == first_id
        # Celery must NOT fire for the duplicate
        mock_task2.delay.assert_not_called()

    async def test_unsupported_extension(self, client: AsyncClient) -> None:
        r = await client.post("/api/v1/uploads", files=_file(b"data", "file.txt"))
        assert r.status_code == 400

    async def test_empty_file(self, client: AsyncClient) -> None:
        r = await client.post("/api/v1/uploads", files=_file(b"", "empty.pdf"))
        assert r.status_code == 400

    async def test_raw_file_written_to_disk(self, client: AsyncClient, tmp_path: Path) -> None:
        content = b"%PDF-1.4 persistent"
        with patch("llm_wiki.api.v1.uploads.process_file_task") as mock_task:
            mock_task.delay.return_value = MagicMock(id="t3")
            r = await client.post("/api/v1/uploads", files=_file(content))

        file_id = r.json()["document_id"]
        raw_files = list((tmp_path / "raw").glob(f"{file_id}.*"))
        assert len(raw_files) == 1
        assert raw_files[0].read_bytes() == content


# ---------------------------------------------------------------------------
# POST /api/v1/materials/upload  — batch
# ---------------------------------------------------------------------------


class TestUploadBatch:
    async def test_two_files(self, client: AsyncClient) -> None:
        files = _files_batch(("alpha.pdf", b"%PDF-1.4 alpha"), ("beta.pdf", b"%PDF-1.4 beta"))
        with patch("llm_wiki.api.v1.uploads.process_file_task") as mock_task:
            mock_task.delay.return_value = MagicMock(id="t")
            r = await client.post("/api/v1/materials/upload", files=files)

        assert r.status_code == 200
        data = r.json()
        assert len(data["uploaded"]) == 2
        assert data["skipped"] == []
        assert mock_task.delay.call_count == 2

    async def test_duplicate_goes_to_skipped(self, client: AsyncClient) -> None:
        content = b"%PDF-1.4 same bytes"

        with patch("llm_wiki.api.v1.uploads.process_file_task") as mock_task:
            mock_task.delay.return_value = MagicMock(id="t1")
            r1 = await client.post("/api/v1/uploads", files=_file(content, "first.pdf"))
        first_id = r1.json()["document_id"]

        # Batch upload: same content + a fresh file
        files = _files_batch(
            ("dup.pdf", content),
            ("new.pdf", b"%PDF-1.4 new content"),
        )
        with patch("llm_wiki.api.v1.uploads.process_file_task") as mock_task2:
            mock_task2.delay.return_value = MagicMock(id="t2")
            r2 = await client.post("/api/v1/materials/upload", files=files)

        assert r2.status_code == 200
        data = r2.json()
        assert len(data["uploaded"]) == 1
        assert len(data["skipped"]) == 1
        assert data["skipped"][0]["document_id"] == first_id
        assert data["skipped"][0]["status"] == "duplicate"
        # Only one new task enqueued (for "new.pdf")
        assert mock_task2.delay.call_count == 1

    async def test_empty_files_list(self, client: AsyncClient) -> None:
        # Sending no files at all should fail with 422 (required field)
        r = await client.post("/api/v1/materials/upload")
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/v1/documents/{id}/status  — polling
# ---------------------------------------------------------------------------


class TestDocumentStatus:
    async def _upload_and_get_id(self, client: AsyncClient) -> str:
        with patch("llm_wiki.api.v1.uploads.process_file_task") as mock_task:
            mock_task.delay.return_value = MagicMock(id="t")
            r = await client.post("/api/v1/uploads", files=_file())
        return r.json()["document_id"]

    async def test_queued_after_upload(self, client: AsyncClient) -> None:
        doc_id = await self._upload_and_get_id(client)
        r = await client.get(f"/api/v1/documents/{doc_id}/status")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "queued"
        assert data["progress"] == 0

    async def test_done_after_status_update(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        doc_id = await self._upload_and_get_id(client)

        # Simulate the pipeline completing
        await db_session.execute(
            sa_update(FileRecord)
            .where(FileRecord.file_id == doc_id)
            .values(status="DONE")
        )
        await db_session.commit()

        r = await client.get(f"/api/v1/documents/{doc_id}/status")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "done"
        assert data["progress"] == 100

    async def test_processing_at_50_for_searched(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        doc_id = await self._upload_and_get_id(client)

        await db_session.execute(
            sa_update(FileRecord)
            .where(FileRecord.file_id == doc_id)
            .values(status="SEARCHED")
        )
        await db_session.commit()

        r = await client.get(f"/api/v1/documents/{doc_id}/status")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "processing"
        assert data["progress"] == 50

    async def test_error_on_failed(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        doc_id = await self._upload_and_get_id(client)

        await db_session.execute(
            sa_update(FileRecord)
            .where(FileRecord.file_id == doc_id)
            .values(status="FAILED")
        )
        await db_session.commit()

        r = await client.get(f"/api/v1/documents/{doc_id}/status")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "error"
        assert data["progress"] == 0

    async def test_404_missing(self, client: AsyncClient) -> None:
        r = await client.get("/api/v1/documents/nonexistent/status")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /api/v1/documents/{id}/sources/{source_id}
# ---------------------------------------------------------------------------


class TestDeleteSource:
    async def _upload_and_get_id(self, client: AsyncClient) -> str:
        with patch("llm_wiki.api.v1.uploads.process_file_task") as mock_task:
            mock_task.delay.return_value = MagicMock(id="t")
            r = await client.post("/api/v1/uploads", files=_file())
        return r.json()["document_id"]

    async def test_403_non_admin(self, client: AsyncClient) -> None:
        doc_id = await self._upload_and_get_id(client)
        r = await client.delete(
            f"/api/v1/documents/{doc_id}/sources/{doc_id}",
            headers={"X-User-Role": "employee"},
        )
        assert r.status_code == 403

    async def test_200_admin_soft_deletes(self, client: AsyncClient) -> None:
        doc_id = await self._upload_and_get_id(client)
        r = await client.delete(
            f"/api/v1/documents/{doc_id}/sources/{doc_id}",
            headers={"X-User-Role": "admin"},
        )
        assert r.status_code == 200
        # Document is now deleted
        r2 = await client.get(f"/api/v1/documents/{doc_id}")
        assert r2.status_code == 404

    async def test_404_wrong_source_id(self, client: AsyncClient) -> None:
        doc_id = await self._upload_and_get_id(client)
        r = await client.delete(
            f"/api/v1/documents/{doc_id}/sources/wrong-id",
            headers={"X-User-Role": "admin"},
        )
        assert r.status_code == 404

    async def test_404_missing_document(self, client: AsyncClient) -> None:
        r = await client.delete(
            "/api/v1/documents/nonexistent/sources/nonexistent",
            headers={"X-User-Role": "admin"},
        )
        assert r.status_code == 404
