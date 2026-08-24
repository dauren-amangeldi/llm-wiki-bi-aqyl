"""BUG-24: «Мои заметки» живут на сервере, приватно для автора."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llm_wiki.api.deps import get_db
from llm_wiki.main import app

ALICE = {"X-User-Email": "alice@bi.group"}
BOB = {"X-User-Email": "bob@bi.group"}


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


async def test_note_roundtrip_and_update(client: AsyncClient, db_session) -> None:
    # Нет заметки — пустой текст, не 404.
    resp = await client.get("/api/v1/notes/case-n1", headers=ALICE)
    assert resp.status_code == 200 and resp.json()["text"] == ""

    assert (
        await client.put("/api/v1/notes/case-n1", json={"text": "Первая мысль"}, headers=ALICE)
    ).status_code == 200
    assert (await client.get("/api/v1/notes/case-n1", headers=ALICE)).json()["text"] == "Первая мысль"

    # Upsert перезаписывает.
    await client.put("/api/v1/notes/case-n1", json={"text": "Вторая версия"}, headers=ALICE)
    assert (await client.get("/api/v1/notes/case-n1", headers=ALICE)).json()["text"] == "Вторая версия"


async def test_note_is_private_per_owner(client: AsyncClient, db_session) -> None:
    await client.put("/api/v1/notes/doc-77", json={"text": "секрет Алисы"}, headers=ALICE)
    assert (await client.get("/api/v1/notes/doc-77", headers=BOB)).json()["text"] == ""
    # И у Боба своя заметка на тот же документ, независимая.
    await client.put("/api/v1/notes/doc-77", json={"text": "заметка Боба"}, headers=BOB)
    assert (await client.get("/api/v1/notes/doc-77", headers=ALICE)).json()["text"] == "секрет Алисы"


async def test_empty_text_deletes_note(client: AsyncClient, db_session) -> None:
    await client.put("/api/v1/notes/doc-9", json={"text": "будет стёрто"}, headers=ALICE)
    await client.put("/api/v1/notes/doc-9", json={"text": "   "}, headers=ALICE)
    assert (await client.get("/api/v1/notes/doc-9", headers=ALICE)).json()["text"] == ""
