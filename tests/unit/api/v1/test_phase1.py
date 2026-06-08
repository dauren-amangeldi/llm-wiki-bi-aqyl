"""Phase 1 integration tests — GET/DELETE /documents, /sources, /dossier, /raw.

Test strategy mirrors tests/unit/api/test_lw16_endpoints.py:
  - In-memory SQLite, ``get_db`` dependency overridden.
  - ``settings.data_dir`` redirected to ``tmp_path``.
  - httpx AsyncClient drives the real ASGI app.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from llm_wiki.api.deps import get_db
from llm_wiki.config import settings
from llm_wiki.main import app
from llm_wiki.storage.metadata import Base, FileRecord


# ---------------------------------------------------------------------------
# Fixtures
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


async def _insert(
    db_session: AsyncSession,
    file_id: str = "file-001",
    original_name: str = "my-doc.pdf",
    status: str = "DONE",
) -> FileRecord:
    record = FileRecord(
        file_id=file_id,
        original_name=original_name,
        status=status,
        created_pages=[],
        updated_pages=[],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db_session.add(record)
    await db_session.commit()
    await db_session.refresh(record)
    return record


# ---------------------------------------------------------------------------
# GET /api/v1/documents
# ---------------------------------------------------------------------------


class TestListDocuments:
    async def test_empty(self, client: AsyncClient) -> None:
        r = await client.get("/api/v1/documents")
        assert r.status_code == 200
        assert r.json() == []

    async def test_returns_one_material(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        await _insert(db_session)
        r = await client.get("/api/v1/documents")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 1
        m = data[0]
        assert m["document_id"] == "file-001"
        assert m["title"] == "my-doc"
        assert m["content_type"] == "pdf"
        assert m["status"] == "DONE"
        assert m["scope"] == "internal"
        assert m["title_i18n"] == {"ru": "my-doc"}

    async def test_deleted_excluded(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        await _insert(db_session, status="deleted")
        r = await client.get("/api/v1/documents")
        assert r.status_code == 200
        assert r.json() == []

    async def test_q_filter_match(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        await _insert(db_session, file_id="a", original_name="alpha.pdf")
        await _insert(db_session, file_id="b", original_name="beta.pdf")
        r = await client.get("/api/v1/documents?q=alp")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 1
        assert data[0]["title"] == "alpha"

    async def test_q_filter_no_match(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        await _insert(db_session)
        r = await client.get("/api/v1/documents?q=zzz")
        assert r.status_code == 200
        assert r.json() == []

    async def test_md_file_content_type(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        await _insert(db_session, original_name="note.md")
        r = await client.get("/api/v1/documents")
        assert r.status_code == 200
        assert r.json()[0]["content_type"] == "markdown"


# ---------------------------------------------------------------------------
# GET /api/v1/documents/{document_id}
# ---------------------------------------------------------------------------


class TestGetDocument:
    async def test_404_missing(self, client: AsyncClient) -> None:
        r = await client.get("/api/v1/documents/nonexistent")
        assert r.status_code == 404

    async def test_404_deleted(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        await _insert(db_session, status="deleted")
        r = await client.get("/api/v1/documents/file-001")
        assert r.status_code == 404

    async def test_200_happy(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        await _insert(db_session)
        r = await client.get("/api/v1/documents/file-001")
        assert r.status_code == 200
        assert r.json()["document_id"] == "file-001"

    async def test_snippet_from_wiki(
        self, client: AsyncClient, db_session: AsyncSession, tmp_path: Path
    ) -> None:
        await _insert(db_session, original_name="my-doc.pdf")
        (tmp_path / "wiki" / "my-doc.md").write_text("Hello wiki content!", encoding="utf-8")
        r = await client.get("/api/v1/documents/file-001")
        assert r.status_code == 200
        assert r.json()["snippet"] == "Hello wiki content!"

    async def test_snippet_truncated_at_200(
        self, client: AsyncClient, db_session: AsyncSession, tmp_path: Path
    ) -> None:
        await _insert(db_session, original_name="my-doc.pdf")
        (tmp_path / "wiki" / "my-doc.md").write_text("X" * 300, encoding="utf-8")
        r = await client.get("/api/v1/documents/file-001")
        assert r.status_code == 200
        assert r.json()["snippet"] == "X" * 200

    async def test_no_wiki_snippet_is_none(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        await _insert(db_session)
        r = await client.get("/api/v1/documents/file-001")
        assert r.status_code == 200
        assert r.json()["snippet"] is None


# ---------------------------------------------------------------------------
# DELETE /api/v1/documents/{document_id}
# ---------------------------------------------------------------------------


class TestDeleteDocument:
    async def test_403_non_admin(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        await _insert(db_session)
        r = await client.delete(
            "/api/v1/documents/file-001",
            headers={"X-User-Role": "employee"},
        )
        assert r.status_code == 403

    async def test_403_default_user(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        # Default get_current_user injects role="employee" when no header given
        await _insert(db_session)
        r = await client.delete("/api/v1/documents/file-001")
        assert r.status_code == 403

    async def test_200_admin_soft_deletes(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        await _insert(db_session)
        r = await client.delete(
            "/api/v1/documents/file-001",
            headers={"X-User-Role": "admin"},
        )
        assert r.status_code == 200
        # Record is now soft-deleted — GET returns 404
        r2 = await client.get("/api/v1/documents/file-001")
        assert r2.status_code == 404

    async def test_404_missing(self, client: AsyncClient) -> None:
        r = await client.delete(
            "/api/v1/documents/nonexistent",
            headers={"X-User-Role": "admin"},
        )
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/v1/documents/{document_id}/sources
# ---------------------------------------------------------------------------


class TestSources:
    async def test_404_missing(self, client: AsyncClient) -> None:
        r = await client.get("/api/v1/documents/nonexistent/sources")
        assert r.status_code == 404

    async def test_returns_one_source(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        await _insert(db_session)
        r = await client.get("/api/v1/documents/file-001/sources")
        assert r.status_code == 200
        data = r.json()
        assert "items" in data
        assert len(data["items"]) == 1
        item = data["items"][0]
        assert item["title"] == "my-doc.pdf"
        assert item["content_type"] == "pdf"
        assert item["path"] == "/api/v1/files/file-001/raw"
        assert item["document_id"] == "file-001"


# ---------------------------------------------------------------------------
# GET /api/v1/documents/{document_id}/dossier
# ---------------------------------------------------------------------------


class TestDossier:
    async def test_404_missing(self, client: AsyncClient) -> None:
        r = await client.get("/api/v1/documents/nonexistent/dossier")
        assert r.status_code == 404

    async def test_summary_none_when_no_wiki(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        await _insert(db_session)
        r = await client.get("/api/v1/documents/file-001/dossier")
        assert r.status_code == 200
        assert r.json()["summary"] is None

    async def test_summary_first_500_chars(
        self, client: AsyncClient, db_session: AsyncSession, tmp_path: Path
    ) -> None:
        await _insert(db_session, original_name="my-doc.pdf")
        (tmp_path / "wiki" / "my-doc.md").write_text("A" * 600, encoding="utf-8")
        r = await client.get("/api/v1/documents/file-001/dossier")
        assert r.status_code == 200
        assert r.json()["summary"] == "A" * 500

    async def test_status_propagated(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        await _insert(db_session, status="PROCESSING")
        r = await client.get("/api/v1/documents/file-001/dossier")
        assert r.status_code == 200
        assert r.json()["status"] == "PROCESSING"


# ---------------------------------------------------------------------------
# GET /api/v1/files/{file_id}/raw
# ---------------------------------------------------------------------------


class TestRaw:
    async def test_404_file_not_on_disk(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        await _insert(db_session)
        r = await client.get("/api/v1/files/file-001/raw")
        assert r.status_code == 404

    async def test_serves_pdf(
        self, client: AsyncClient, db_session: AsyncSession, tmp_path: Path
    ) -> None:
        await _insert(db_session)
        (tmp_path / "raw" / "file-001.pdf").write_bytes(b"%PDF-1.4 test")
        r = await client.get("/api/v1/files/file-001/raw")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/pdf")
        assert r.content == b"%PDF-1.4 test"

    async def test_serves_markdown(
        self, client: AsyncClient, db_session: AsyncSession, tmp_path: Path
    ) -> None:
        await _insert(db_session, original_name="note.md")
        (tmp_path / "raw" / "file-001.md").write_bytes(b"# Hello\n")
        r = await client.get("/api/v1/files/file-001/raw")
        assert r.status_code == 200
        assert "text/markdown" in r.headers["content-type"]

    async def test_400_path_traversal(self, client: AsyncClient) -> None:
        r = await client.get("/api/v1/files/..%2Fetc%2Fpasswd/raw")
        assert r.status_code in (400, 422, 404)


# ---------------------------------------------------------------------------
# GET /api/v1/tags  — stub
# ---------------------------------------------------------------------------


class TestTags:
    async def test_returns_empty_list(self, client: AsyncClient) -> None:
        r = await client.get("/api/v1/tags")
        assert r.status_code == 200
        assert r.json() == []
