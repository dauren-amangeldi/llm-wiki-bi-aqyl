"""Endpoint tests for POST /api/v1/advisor/consultations/{id}/confirm."""

from __future__ import annotations

import json
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


def _fake_brief():
    from llm_wiki.api.schemas import DecisionBrief

    return DecisionBrief(
        recommendation="Ограниченное масштабирование",
        why_now="Пилот дал сигнал",
        problem_frame="Масштабировать сразу или проверить переносимость",
        key_assumption="Эффект сохранится",
        rationale="Проверяет переносимость",
        alternatives=["Полное масштабирование"],
        risks=["Другая операционная модель"],
        first_step="Выбрать два подразделения",
        reconsider_if=["Эффект ниже 20%"],
        evidence_strength="medium",
        assumptions=["Пилот репрезентативен"],
        sources=["Отчёт по пилоту"],
    )


async def test_confirm_streams_progress_then_decision_brief(client: AsyncClient) -> None:
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

    with patch("llm_wiki.api.v1.consultations.run_synthesis", new=AsyncMock(return_value=_fake_brief())):
        resp = await client.post(
            f"/api/v1/advisor/consultations/{session_id}/confirm",
            headers={"X-User-Email": "alice@bi.group"},
        )
    assert resp.status_code == 200
    lines = [l for l in resp.text.strip().splitlines() if l.startswith("data: ")]
    events = [json.loads(l.removeprefix("data: ")) for l in lines]
    assert events[0]["status"] == "searching"
    assert events[-1]["mode"] == "decision_brief"
    assert events[-1]["brief"]["recommendation"].startswith("Ограниченное")

    from llm_wiki.storage.metadata import get_advisor_session

    session_factory = app.dependency_overrides[get_db]
    async for db in session_factory():
        row = await get_advisor_session(db, session_id)
        break
    assert row.state == "completed"
