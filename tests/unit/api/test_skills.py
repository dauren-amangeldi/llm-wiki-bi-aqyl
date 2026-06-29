"""API tests for GET/PUT /api/v1/skills (LW-N12)."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llm_wiki.api.deps import get_db
from llm_wiki.main import app
from llm_wiki.storage.metadata import seed_skills


@pytest_asyncio.fixture
async def client(db_engine) -> AsyncGenerator[AsyncClient, None]:
    factory = async_sessionmaker(bind=db_engine, expire_on_commit=False, autoflush=False)

    async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with factory() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db

    async with factory() as session:
        await seed_skills(session)

    async with AsyncClient(
        transport=ASGITransport(app=app),  # type: ignore[arg-type]
        base_url="http://test",
    ) as ac:
        yield ac

    app.dependency_overrides.clear()


async def test_list_skills_returns_seeded_roles(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/skills")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 10
    slugs = {item["slug"] for item in data}
    assert "modes/advisor" in slugs
    assert "positions/pm" in slugs
    assert all("content" in item and "name" in item for item in data)


async def test_put_skill_updates_prompt(client: AsyncClient) -> None:
    new_prompt = "Updated advisor persona — focus on risk mitigation."
    resp = await client.put(
        "/api/v1/skills/modes/advisor",
        json={"content": new_prompt},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["content"] == new_prompt
    assert body["slug"] == "modes/advisor"

    get_resp = await client.get("/api/v1/skills")
    advisor = next(item for item in get_resp.json() if item["role"] == "advisor")
    assert advisor["content"] == new_prompt


async def test_put_unknown_skill_returns_404(client: AsyncClient) -> None:
    resp = await client.put(
        "/api/v1/skills/positions/unknown",
        json={"content": "nope"},
    )
    assert resp.status_code == 404
