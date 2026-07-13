"""Endpoint tests for PUT /api/v1/advisor/consultations/{id}/snapshot."""

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


async def test_snapshot_update_changes_only_provided_fields(client: AsyncClient) -> None:
    from llm_wiki.agents.consultation import DiscoveryResult

    with patch(
        "llm_wiki.api.v1.consultations.run_discovery",
        new=AsyncMock(return_value=DiscoveryResult(decision_type="market_entry", sufficient_context=True, questions=[])),
    ):
        start = await client.post(
            "/api/v1/advisor/consultations",
            json={"query": "Выходить ли на рынок Алматы?"},
            headers={"X-User-Email": "alice@bi.group"},
        )
    session_id = start.json()["session_id"]
    original_horizon = start.json()["snapshot"]["horizon"]

    resp = await client.put(
        f"/api/v1/advisor/consultations/{session_id}/snapshot",
        json={"decision": "Уточнённая формулировка решения"},
        headers={"X-User-Email": "alice@bi.group"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["snapshot"]["decision"] == "Уточнённая формулировка решения"
    assert body["snapshot"]["horizon"] == original_horizon  # не переданное поле не изменилось


async def test_snapshot_update_404_for_unknown_session(client: AsyncClient) -> None:
    resp = await client.put(
        "/api/v1/advisor/consultations/does-not-exist/snapshot",
        json={"decision": "X"},
        headers={"X-User-Email": "alice@bi.group"},
    )
    assert resp.status_code == 404
