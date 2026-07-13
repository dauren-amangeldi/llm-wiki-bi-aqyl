"""Endpoint tests for POST /api/v1/advisor/consultations."""

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


async def test_start_returns_clarification_when_context_insufficient(client: AsyncClient) -> None:
    from llm_wiki.agents.consultation import DiscoveryResult
    from llm_wiki.api.schemas import ClarificationQuestion

    fake_result = DiscoveryResult(
        decision_type="initiative_scaling",
        sufficient_context=False,
        questions=[ClarificationQuestion(id="q1", text="Какой результат важнее?", options=["A", "B"])],
    )
    with patch("llm_wiki.api.v1.consultations.run_discovery", new=AsyncMock(return_value=fake_result)):
        resp = await client.post(
            "/api/v1/advisor/consultations",
            json={"query": "Стоит ли масштабировать пилот на весь холдинг?"},
            headers={"X-User-Email": "alice@bi.group"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "clarification_required"
    assert body["decision_type"] == "initiative_scaling"
    assert body["questions"][0]["id"] == "q1"
    assert body["session_id"].startswith("advisor-session-")


async def test_start_skips_straight_to_snapshot_when_context_sufficient(client: AsyncClient) -> None:
    from llm_wiki.agents.consultation import DiscoveryResult

    fake_result = DiscoveryResult(decision_type="market_entry", sufficient_context=True, questions=[])
    with patch("llm_wiki.api.v1.consultations.run_discovery", new=AsyncMock(return_value=fake_result)):
        resp = await client.post(
            "/api/v1/advisor/consultations",
            json={"query": "Выходить ли на рынок Алматы?"},
            headers={"X-User-Email": "alice@bi.group"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "understanding_snapshot"
    assert body["snapshot"]["decision"]
