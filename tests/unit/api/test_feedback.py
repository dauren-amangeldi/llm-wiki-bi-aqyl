"""BUG-08/09: фидбэк сохраняется на сервере (петля качества)."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llm_wiki.api.deps import get_db
from llm_wiki.main import app
from llm_wiki.storage.metadata import FeedbackRecord


@pytest_asyncio.fixture
async def client(db_engine) -> AsyncGenerator[AsyncClient, None]:
    factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
        bind=db_engine, expire_on_commit=False, autoflush=False
    )

    async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with factory() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db
    async with AsyncClient(
        transport=ASGITransport(app=app),  # type: ignore[arg-type]
        base_url="http://test",
    ) as c:
        yield c
    app.dependency_overrides.clear()


async def test_vote_and_report_are_persisted(client: AsyncClient, db_session) -> None:
    resp = await client.post(
        "/api/v1/feedback",
        json={"entity_type": "chat_answer", "entity_id": "case-1", "vote": "up",
              "comment": "Хороший ответ про BATNA"},
        headers={"X-User-Email": "alice@bi.group"},
    )
    assert resp.status_code == 201

    resp = await client.post(
        "/api/v1/feedback",
        json={"entity_type": "chat_answer_report", "entity_id": "case-1",
              "reason": "report_reason_wrong", "comment": "Ответ противоречит документу"},
        headers={"X-User-Email": "alice@bi.group"},
    )
    assert resp.status_code == 201

    rows = (await db_session.scalars(select(FeedbackRecord))).all()
    assert len(rows) == 2
    votes = {r.vote for r in rows}
    assert "up" in votes
    report = next(r for r in rows if r.entity_type == "chat_answer_report")
    assert report.reason == "report_reason_wrong"
    assert report.owner == "alice@bi.group"


async def test_invalid_vote_rejected(client: AsyncClient, db_session) -> None:
    resp = await client.post(
        "/api/v1/feedback",
        json={"entity_type": "chat_answer", "entity_id": "x", "vote": "maybe"},
    )
    assert resp.status_code == 422
