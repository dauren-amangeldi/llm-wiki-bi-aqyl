"""Unit tests for advisor session persistence via POST /api/v1/advisor."""

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
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"  # type: ignore[arg-type]
    ) as ac:
        yield ac
    app.dependency_overrides.pop(get_db, None)


def _fake_advise_result():
    from llm_wiki.agents.advisor import AdvisorPoint, AdvisorResponse

    return AdvisorResponse(
        title="Строить или нет",
        summary="Рынок готов",
        points=[AdvisorPoint(heading="Спрос растёт", body="...", metric="", tag="", case_id="")],
        source="rag",
        caseCount=3,
        cost_usd=0.001,
        refusal=False,
        refusal_message="",
    )


async def test_advisor_without_session_id_creates_one_and_returns_it(client: AsyncClient) -> None:
    with patch("llm_wiki.agents.advisor.AdvisorAgent.advise", new=AsyncMock(return_value=_fake_advise_result())):
        resp = await client.post(
            "/api/v1/advisor",
            json={"query": "Стоит ли строить в KZ?", "role": "employee", "language": "ru"},
            headers={"X-User-Email": "alice@bi.group"},
        )
    assert resp.status_code == 200
    event = json.loads(resp.text.strip().splitlines()[-1].removeprefix("data: "))
    assert event["session_id"].startswith("advisor-session-")


async def test_advisor_with_session_id_reuses_it_and_persists_turns(client: AsyncClient) -> None:
    from llm_wiki.storage.metadata import create_advisor_session, list_chat_messages

    session_factory = app.dependency_overrides[get_db]
    async for db in session_factory():
        created = await create_advisor_session(db, user_key="alice@bi.group", title="Existing")
        break

    with patch("llm_wiki.agents.advisor.AdvisorAgent.advise", new=AsyncMock(return_value=_fake_advise_result())):
        resp = await client.post(
            "/api/v1/advisor",
            json={
                "query": "А если конкретно про Алматы?",
                "role": "employee",
                "language": "ru",
                "session_id": created.id,
            },
            headers={"X-User-Email": "alice@bi.group"},
        )
    assert resp.status_code == 200
    event = json.loads(resp.text.strip().splitlines()[-1].removeprefix("data: "))
    assert event["session_id"] == created.id

    async for db in session_factory():
        msgs = await list_chat_messages(db, user_key="alice@bi.group", scope_type="advisor", scope_id=created.id)
        break
    assert [m.role for m in msgs] == ["user", "assistant"]
    assert msgs[0].text == "А если конкретно про Алматы?"


async def test_list_advisor_sessions_returns_only_the_caller_s_sessions(client: AsyncClient) -> None:
    from llm_wiki.storage.metadata import create_advisor_session

    session_factory = app.dependency_overrides[get_db]
    async for db in session_factory():
        await create_advisor_session(db, user_key="alice@bi.group", title="Mine")
        await create_advisor_session(db, user_key="bob@bi.group", title="Not mine")
        break

    resp = await client.get("/api/v1/advisor/sessions", headers={"X-User-Email": "alice@bi.group"})
    assert resp.status_code == 200
    titles = [s["title"] for s in resp.json()]
    assert titles == ["Mine"]


async def test_get_session_messages_returns_404_for_someone_elses_session(client: AsyncClient) -> None:
    from llm_wiki.storage.metadata import create_advisor_session

    session_factory = app.dependency_overrides[get_db]
    async for db in session_factory():
        created = await create_advisor_session(db, user_key="bob@bi.group", title="Bob's")
        break

    resp = await client.get(
        f"/api/v1/advisor/sessions/{created.id}/messages",
        headers={"X-User-Email": "alice@bi.group"},
    )
    assert resp.status_code == 404


async def test_get_session_messages_returns_persisted_turns(client: AsyncClient) -> None:
    from llm_wiki.storage.metadata import append_chat_message, create_advisor_session

    session_factory = app.dependency_overrides[get_db]
    async for db in session_factory():
        created = await create_advisor_session(db, user_key="alice@bi.group", title="Mine")
        await append_chat_message(
            db, user_key="alice@bi.group", scope_type="advisor", scope_id=created.id,
            role="user", text_body="Вопрос",
        )
        await append_chat_message(
            db, user_key="alice@bi.group", scope_type="advisor", scope_id=created.id,
            role="assistant", text_body="Ответ",
        )
        break

    resp = await client.get(
        f"/api/v1/advisor/sessions/{created.id}/messages",
        headers={"X-User-Email": "alice@bi.group"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert [m["role"] for m in body] == ["user", "assistant"]
    assert body[0]["text"] == "Вопрос"
