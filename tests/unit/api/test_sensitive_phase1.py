"""Phase 1 of sensitive files: owner/sensitive capture on files and cases."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llm_wiki.api.deps import get_db
from llm_wiki.main import app
from llm_wiki.storage.metadata import create_file_record, get_file_record


@pytest_asyncio.fixture
async def client(db_engine) -> AsyncGenerator[AsyncClient, None]:
    factory = async_sessionmaker(bind=db_engine, expire_on_commit=False, autoflush=False)

    async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with factory() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db
    async with AsyncClient(
        transport=ASGITransport(app=app),  # type: ignore[arg-type]
        base_url="http://test",
    ) as ac:
        yield ac
    app.dependency_overrides.clear()


async def test_create_file_record_persists_sensitive_and_owner(
    db_session: AsyncSession,
) -> None:
    await create_file_record(
        db_session, "sf-1", "secret.md", sensitive=True, owner="alice@bi.group"
    )
    db_session.expire_all()
    rec = await get_file_record(db_session, "sf-1")
    assert rec is not None
    assert rec.sensitive is True
    assert rec.owner == "alice@bi.group"


async def test_create_file_record_defaults_non_sensitive(db_session: AsyncSession) -> None:
    await create_file_record(db_session, "sf-2", "public.md")
    db_session.expire_all()
    rec = await get_file_record(db_session, "sf-2")
    assert rec is not None
    assert rec.sensitive is False
    assert rec.owner is None


async def test_create_case_persists_privacy(client: AsyncClient) -> None:
    # Private case
    r = await client.post(
        "/api/v1/cases",
        json={"title": "HR docs", "sensitive": True},
        headers={"X-User-Email": "alice@bi.group"},
    )
    assert r.status_code == 201
    assert r.json()["sensitive"] is True

    # Public case
    r2 = await client.post(
        "/api/v1/cases",
        json={"title": "Public", "sensitive": False},
        headers={"X-User-Email": "alice@bi.group"},
    )
    assert r2.status_code == 201
    assert r2.json()["sensitive"] is False

    # Listing carries the flag
    lst = await client.get("/api/v1/cases", headers={"X-User-Email": "alice@bi.group"})
    by_title = {c["title"]: c["sensitive"] for c in lst.json()}
    assert by_title["HR docs"] is True
    assert by_title["Public"] is False
