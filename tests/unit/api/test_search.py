"""Smoke tests for GET /api/v1/search (document name search)."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from llm_wiki.api.deps import get_db
from llm_wiki.main import app
from llm_wiki.storage.metadata import Base, create_file_record, update_file_status


@pytest_asyncio.fixture
async def db_engine():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def client(db_engine) -> AsyncGenerator[AsyncClient, None]:
    factory = async_sessionmaker(bind=db_engine, expire_on_commit=False, autoflush=False)

    async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with factory() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db

    async with factory() as session:
        await create_file_record(session, "doc-1", "Sales Report Q1.pdf")
        await create_file_record(session, "doc-2", "Marketing Plan.md")
        await update_file_status(session, "doc-1", "DONE")
        await update_file_status(session, "doc-2", "DONE")

    async with AsyncClient(
        transport=ASGITransport(app=app),  # type: ignore[arg-type]
        base_url="http://test",
    ) as ac:
        yield ac

    app.dependency_overrides.clear()


async def test_search_returns_document_hits(client: AsyncClient) -> None:
    """GET /search returns document hits matching filename."""
    resp = await client.get("/api/v1/search", params={"q": "Sales"})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["document_id"] == "doc-1"
    assert data[0]["document_title"] == "Sales Report Q1"
    assert data[0]["scope"] == "internal"
    assert data[0]["content_type"] == "pdf"


async def test_search_empty_query_returns_empty(client: AsyncClient) -> None:
    """Empty query returns an empty list."""
    resp = await client.get("/api/v1/search", params={"q": ""})
    assert resp.status_code == 200
    assert resp.json() == []


async def test_search_accepts_query_alias(client: AsyncClient) -> None:
    """Frontend may send `query` instead of `q`."""
    resp = await client.get("/api/v1/search", params={"query": "Marketing"})
    assert resp.status_code == 200
    assert len(resp.json()) == 1
    assert resp.json()[0]["document_id"] == "doc-2"
