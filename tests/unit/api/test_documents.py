"""Smoke tests for document bridge endpoints."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llm_wiki.api.deps import get_db
from llm_wiki.config import settings
from llm_wiki.main import app
from llm_wiki.storage.metadata import create_file_record, update_file_status


@pytest_asyncio.fixture
async def client(
    db_engine,
    tmp_path: Path,
) -> AsyncGenerator[AsyncClient, None]:
    data_dir = tmp_path / "data"
    wiki_dir = data_dir / "wiki"
    wiki_dir.mkdir(parents=True)
    (wiki_dir / "sample-page.md").write_text("# Sample\n\nBody text.", encoding="utf-8")
    object.__setattr__(settings, "data_dir", data_dir)

    factory = async_sessionmaker(bind=db_engine, expire_on_commit=False, autoflush=False)

    async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with factory() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db

    async with factory() as session:
        record = await create_file_record(session, "doc-1", "Report.md")
        record.created_pages = ["sample-page"]
        await session.commit()
        await update_file_status(session, "doc-1", "DONE")

    async with AsyncClient(
        transport=ASGITransport(app=app),  # type: ignore[arg-type]
        base_url="http://test",
    ) as ac:
        yield ac

    app.dependency_overrides.pop(get_db, None)


async def test_get_document_text(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/documents/doc-1/text")
    assert resp.status_code == 200
    body = resp.json()
    assert body["slug"] == "sample-page"
    assert "Sample" in (body["content"] or "")


async def test_get_document_tags(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/documents/doc-1/tags")
    assert resp.status_code == 200
    body = resp.json()
    assert body["tags"] == []
    assert len(body["suggestions"]) == 4


async def test_get_document_sources(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/documents/doc-1/sources")
    assert resp.status_code == 200
    assert len(resp.json()) == 1
    assert resp.json()[0]["document_id"] == "doc-1"
