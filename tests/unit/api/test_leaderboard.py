"""Unit tests for GET /api/v1/leaderboard/cases."""

from __future__ import annotations

from collections.abc import AsyncGenerator

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
        transport=ASGITransport(app=app),  # type: ignore[arg-type]
        base_url="http://test",
    ) as ac:
        yield ac

    app.dependency_overrides.pop(get_db, None)


async def test_leaderboard_empty_when_no_cases(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/leaderboard/cases")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_leaderboard_ranks_by_case_count(client: AsyncClient) -> None:
    await client.post(
        "/api/v1/cases",
        json={"id": "c1", "title": "Alpha"},
        headers={"X-User-Email": "alice@bi.group"},
    )
    await client.post(
        "/api/v1/cases",
        json={"id": "c2", "title": "Beta"},
        headers={"X-User-Email": "alice@bi.group"},
    )
    await client.post(
        "/api/v1/cases",
        json={"id": "c3", "title": "Gamma"},
        headers={"X-User-Email": "bob@bi.group"},
    )

    resp = await client.get("/api/v1/leaderboard/cases")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 2
    assert body[0]["user_id"] == "alice@bi.group"
    assert body[0]["case_count"] == 2
    assert body[1]["user_id"] == "bob@bi.group"
    assert body[1]["case_count"] == 1


async def test_leaderboard_excludes_unattributed_cases(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """A case created before owner_id existed (owner_id IS NULL) is not
    attributable to anyone and must not appear as a phantom leaderboard row.

    Note: this uses the injected ``db_session`` fixture (bound to the
    per-test ``db_engine``/``TEST_DATABASE_URL``) rather than
    ``llm_wiki.api.deps._SessionLocal`` — in this repo, ``_SessionLocal`` is
    bound at import time to ``settings.database_url``, which resolves to the
    live application database (DATABASE_URL=.../llmwiki) even under pytest,
    not to the isolated test database. Inserting via ``_SessionLocal`` here
    would write into the dev DB while the ``client`` fixture's overridden
    ``get_db`` queries the separate test DB, so the row would never be seen
    by the endpoint under test (see test_access_control.py's `gate_client`
    fixture, which explicitly monkeypatches `deps._SessionLocal` for the same
    reason).
    """
    from datetime import UTC, datetime

    from llm_wiki.storage.metadata import CaseRecord

    db_session.add(
        CaseRecord(
            id="c-legacy",
            title="Legacy case",
            doc_ids=[],
            owner_id=None,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
    )
    await db_session.commit()

    resp = await client.get("/api/v1/leaderboard/cases")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_leaderboard_respects_limit(client: AsyncClient) -> None:
    for i, email in enumerate(["a@bi.group", "b@bi.group", "c@bi.group"]):
        await client.post(
            "/api/v1/cases",
            json={"id": f"case-{i}", "title": f"Case {i}"},
            headers={"X-User-Email": email},
        )

    resp = await client.get("/api/v1/leaderboard/cases?limit=2")
    assert resp.status_code == 200
    assert len(resp.json()) == 2
