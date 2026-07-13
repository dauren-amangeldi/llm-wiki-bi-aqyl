"""Endpoint tests for POST /api/v1/advisor/consultations/{id}/respond."""

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


async def test_respond_builds_snapshot_and_advances_state(client: AsyncClient, db_engine) -> None:
    from llm_wiki.agents.consultation import DiscoveryResult
    from llm_wiki.api.schemas import ClarificationQuestion, UnderstandingSnapshot
    from llm_wiki.storage.metadata import get_advisor_session

    fake_discovery = DiscoveryResult(
        decision_type="initiative_scaling",
        sufficient_context=False,
        questions=[ClarificationQuestion(id="q1", text="Какой результат важнее?", options=["A", "B"])],
    )
    with patch("llm_wiki.api.v1.consultations.run_discovery", new=AsyncMock(return_value=fake_discovery)):
        start_resp = await client.post(
            "/api/v1/advisor/consultations",
            json={"query": "Стоит ли масштабировать пилот на весь холдинг?"},
            headers={"X-User-Email": "alice@bi.group"},
        )
    assert start_resp.status_code == 200
    session_id = start_resp.json()["session_id"]

    fake_snapshot = UnderstandingSnapshot(
        decision="Масштабировать ли пилот",
        desired_outcome="Сократить срок решений",
        horizon="Текущий год",
        constraints=["Команда ограничена"],
        stakeholders=["ИТ"],
        success_criteria=["Измеримое сокращение"],
        assumptions=["Пилот репрезентативен"],
    )
    with patch("llm_wiki.api.v1.consultations.build_snapshot", new=AsyncMock(return_value=fake_snapshot)):
        respond_resp = await client.post(
            f"/api/v1/advisor/consultations/{session_id}/respond",
            json={"answers": [{"question_id": "q1", "answer": "A", "skipped": False}]},
            headers={"X-User-Email": "alice@bi.group"},
        )
    assert respond_resp.status_code == 200
    body = respond_resp.json()
    assert body["mode"] == "understanding_snapshot"
    assert body["snapshot"]["decision"] == "Масштабировать ли пилот"

    session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
        bind=db_engine, expire_on_commit=False, autoflush=False
    )
    async with session_factory() as db:
        session = await get_advisor_session(db, session_id)
        assert session is not None
        assert session.state == "context_review"


async def test_respond_unknown_session_returns_404(client: AsyncClient) -> None:
    with patch("llm_wiki.api.v1.consultations.build_snapshot", new=AsyncMock()):
        resp = await client.post(
            "/api/v1/advisor/consultations/does-not-exist/respond",
            json={"answers": []},
            headers={"X-User-Email": "alice@bi.group"},
        )
    assert resp.status_code == 404
