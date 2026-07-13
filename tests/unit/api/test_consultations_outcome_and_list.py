"""Endpoint tests for outcome fixation, listing, and resuming consultations."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, patch

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llm_wiki.api.deps import get_db
from llm_wiki.main import app


@pytest_asyncio.fixture
async def client(db_engine) -> AsyncGenerator[AsyncClient, None]:
    session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
        bind=db_engine, expire_on_commit=False, autoflush=False
    )

    async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:  # type: ignore[arg-type]
        yield ac
    app.dependency_overrides.pop(get_db, None)


async def _start(client: AsyncClient, query: str) -> str:
    from llm_wiki.agents.consultation import DiscoveryResult

    with patch(
        "llm_wiki.api.v1.consultations.run_discovery",
        new=AsyncMock(return_value=DiscoveryResult(decision_type="market_entry", sufficient_context=True, questions=[])),
    ):
        resp = await client.post(
            "/api/v1/advisor/consultations", json={"query": query}, headers={"X-User-Email": "alice@bi.group"}
        )
    return resp.json()["session_id"]


async def test_outcome_sets_state(client: AsyncClient) -> None:
    session_id = await _start(client, "Выходить ли на рынок Алматы?")
    resp = await client.post(
        f"/api/v1/advisor/consultations/{session_id}/outcome",
        json={"outcome": "decided"},
        headers={"X-User-Email": "alice@bi.group"},
    )
    assert resp.status_code == 200
    assert resp.json()["outcome"] == "decided"


async def test_list_returns_only_callers_consultations(client: AsyncClient) -> None:
    await _start(client, "Мой вопрос")
    resp = await client.get("/api/v1/advisor/consultations", headers={"X-User-Email": "alice@bi.group"})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["state"] == "context_review"


async def test_resume_returns_current_snapshot(client: AsyncClient) -> None:
    session_id = await _start(client, "Стоит ли строить в Астане?")
    resp = await client.get(f"/api/v1/advisor/consultations/{session_id}", headers={"X-User-Email": "alice@bi.group"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "context_review"
    assert body["snapshot"]["decision"]
