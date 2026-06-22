"""Unit tests for /api/v1/cases CRUD endpoints."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from llm_wiki.api.deps import get_db
from llm_wiki.main import app
from llm_wiki.storage.metadata import Base


@pytest_asyncio.fixture
async def db_engine(tmp_path: Path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/test.db")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def client(db_engine) -> AsyncGenerator[AsyncClient, None]:
    session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
        bind=db_engine, expire_on_commit=False, autoflush=False
    )

    async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db

    async with AsyncClient(
        transport=ASGITransport(app=app),  # type: ignore[arg-type]
        base_url="http://test",
    ) as ac:
        yield ac

    app.dependency_overrides.pop(get_db, None)


async def test_list_cases_empty(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/cases")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_create_case(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/cases",
        json={"title": "My Case", "doc_ids": ["doc-1", "doc-2"]},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["title"] == "My Case"
    assert body["doc_ids"] == ["doc-1", "doc-2"]
    assert "id" in body


async def test_list_cases_after_create(client: AsyncClient) -> None:
    await client.post("/api/v1/cases", json={"title": "Alpha", "doc_ids": []})
    await client.post("/api/v1/cases", json={"title": "Beta", "doc_ids": ["d1"]})

    resp = await client.get("/api/v1/cases")
    assert resp.status_code == 200
    cases = resp.json()
    assert len(cases) == 2
    titles = {c["title"] for c in cases}
    assert titles == {"Alpha", "Beta"}


async def test_update_case(client: AsyncClient) -> None:
    create_resp = await client.post(
        "/api/v1/cases",
        json={"id": "case-upd-1", "title": "Original", "doc_ids": ["d1"]},
    )
    assert create_resp.status_code == 201
    case_id = create_resp.json()["id"]

    put_resp = await client.put(
        f"/api/v1/cases/{case_id}",
        json={"title": "Updated Title", "doc_ids": ["d2", "d3"]},
    )
    assert put_resp.status_code == 200
    assert put_resp.json() == {"ok": True}

    list_resp = await client.get("/api/v1/cases")
    updated = next(c for c in list_resp.json() if c["id"] == case_id)
    assert updated["title"] == "Updated Title"
    assert updated["doc_ids"] == ["d2", "d3"]


async def test_delete_case(client: AsyncClient) -> None:
    create_resp = await client.post(
        "/api/v1/cases",
        json={"id": "case-del-1", "title": "To Delete", "doc_ids": []},
    )
    assert create_resp.status_code == 201
    case_id = create_resp.json()["id"]

    del_resp = await client.delete(f"/api/v1/cases/{case_id}")
    assert del_resp.status_code == 200
    assert del_resp.json() == {"ok": True}

    list_resp = await client.get("/api/v1/cases")
    ids = [c["id"] for c in list_resp.json()]
    assert case_id not in ids


async def test_delete_nonexistent_case_returns_404(client: AsyncClient) -> None:
    resp = await client.delete("/api/v1/cases/does-not-exist")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


async def test_update_nonexistent_case_returns_404(client: AsyncClient) -> None:
    resp = await client.put(
        "/api/v1/cases/ghost-id",
        json={"title": "Ghost", "doc_ids": []},
    )
    assert resp.status_code == 404
