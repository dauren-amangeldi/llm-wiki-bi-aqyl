"""Unit tests for POST /api/v1/feedback (R2-3)."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llm_wiki.api.deps import get_db
from llm_wiki.main import app
from llm_wiki.storage.metadata import Feedback


@pytest_asyncio.fixture
async def session_factory(db_engine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(bind=db_engine, expire_on_commit=False, autoflush=False)


@pytest_asyncio.fixture
async def client(session_factory) -> AsyncGenerator[AsyncClient, None]:
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


async def test_submit_vote_persists_row(client: AsyncClient, session_factory) -> None:
    resp = await client.post(
        "/api/v1/feedback",
        json={"entity_type": "similar_case", "entity_id": "case-a:case-b", "vote": 1},
        headers={"X-User-Email": "user@bi.group"},
    )

    assert resp.status_code == 200
    assert resp.json() == {"ok": True}

    async with session_factory() as s:
        rows = (await s.execute(select(Feedback))).scalars().all()
    assert len(rows) == 1
    assert rows[0].entity_type == "similar_case"
    assert rows[0].vote == 1
    assert rows[0].created_by == "user@bi.group"


async def test_rejects_unknown_entity_type(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/feedback",
        json={"entity_type": "banana", "entity_id": "x", "vote": 1},
    )
    assert resp.status_code == 422


async def test_rejects_invalid_vote(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/feedback",
        json={"entity_type": "twin_verdict", "entity_id": "x", "vote": 5},
    )
    assert resp.status_code == 422
