"""Smoke tests for GET /api/v1/search (LW-N5)."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from llm_wiki.api.deps import get_db
from llm_wiki.main import app
from llm_wiki.storage.metadata import Base
from llm_wiki.storage.wiki_fts import ensure_wiki_fts_table, upsert_wiki_fts


@pytest_asyncio.fixture
async def db_engine():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await ensure_wiki_fts_table(conn)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def client(
    tmp_path: Path,
    db_engine,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncGenerator[AsyncClient, None]:
    factory = async_sessionmaker(bind=db_engine, expire_on_commit=False, autoflush=False)

    async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with factory() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db

    async with factory() as session:
        await upsert_wiki_fts(
            session,
            slug="spinbrush",
            title="SpinBrush",
            body="# SpinBrush\n\nElectric toothbrush product launch details.",
        )

    async with AsyncClient(
        transport=ASGITransport(app=app),  # type: ignore[arg-type]
        base_url="http://test",
    ) as ac:
        yield ac

    app.dependency_overrides.clear()


async def test_search_returns_fts_hits(client: AsyncClient) -> None:
    """GET /search returns wiki hits without LLM."""
    resp = await client.get("/api/v1/search", params={"q": "toothbrush"})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["slug"] == "spinbrush"
    assert data[0]["scope"] == "wiki"
    assert "snippet" in data[0]


async def test_search_short_query_empty(client: AsyncClient) -> None:
    """Queries under 3 characters return an empty list."""
    resp = await client.get("/api/v1/search", params={"q": "ab"})
    assert resp.status_code == 200
    assert resp.json() == []
