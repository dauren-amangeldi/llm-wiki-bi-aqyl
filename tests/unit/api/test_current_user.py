"""Unit tests for get_current_user dependency (LW-N1)."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llm_wiki.api.deps import get_db
from llm_wiki.main import app
from llm_wiki.storage.metadata import get_user_by_id


@pytest_asyncio.fixture
async def client(db_engine) -> AsyncGenerator[AsyncClient, None]:
    """Test client with DB override."""
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


async def test_get_current_user_creates_dev_user_by_default(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Without X-User-Id the stable dev-user is persisted."""
    from llm_wiki.api.deps import get_current_user
    from fastapi import Request

    scope = {"type": "http", "headers": []}
    request = Request(scope)

    user = await get_current_user(request, db_session)
    assert user.id == "dev-user"
    assert user.role == "admin"

    stored = await get_user_by_id(db_session, "dev-user")
    assert stored is not None


async def test_get_current_user_from_header(
    db_session: AsyncSession,
) -> None:
    """X-User-Id header creates and returns a persisted user."""
    from llm_wiki.api.deps import get_current_user
    from fastapi import Request

    scope = {
        "type": "http",
        "headers": [
            (b"x-user-id", b"user-42"),
            (b"x-user-name", b"Test User"),
            (b"x-user-role", b"employee"),
        ],
    }
    request = Request(scope)

    user = await get_current_user(request, db_session)
    assert user.id == "user-42"
    assert user.role == "employee"

    stored = await get_user_by_id(db_session, "user-42")
    assert stored is not None
    assert stored.name == "Test User"
