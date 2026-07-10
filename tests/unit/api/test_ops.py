"""Tests for the hidden ops dashboard endpoint."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llm_wiki.api.deps import get_db
from llm_wiki.config import settings
from llm_wiki.main import app

TOKEN = "test-secret-token"


@pytest_asyncio.fixture
async def client(db_engine, tmp_path: Path) -> AsyncGenerator[AsyncClient, None]:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    object.__setattr__(settings, "data_dir", data_dir)
    object.__setattr__(settings, "ops_dashboard_token", TOKEN)

    factory = async_sessionmaker(bind=db_engine, expire_on_commit=False, autoflush=False)

    async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with factory() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()
    object.__setattr__(settings, "ops_dashboard_token", "")


async def test_missing_token_returns_404(client: AsyncClient) -> None:
    res = await client.get("/api/v1/ops/summary")
    assert res.status_code == 404


async def test_wrong_token_returns_404(client: AsyncClient) -> None:
    res = await client.get("/api/v1/ops/summary", headers={"X-Ops-Token": "wrong"})
    assert res.status_code == 404


async def test_correct_token_aggregates_usage_log(client: AsyncClient) -> None:
    usage_log = settings.usage_log_path
    usage_log.write_text(
        '{"file_id": "f1", "agent_type": "answer", "model": "gpt-5.4-mini", '
        '"input_tokens": 100, "output_tokens": 50, "cached_input_tokens": 0, '
        '"cost_usd": 0.01, "timestamp": "2026-07-10T10:00:00+00:00", "duration_ms": 800}\n'
        '{"file_id": "f2", "agent_type": "search", "model": "gpt-5.4", '
        '"input_tokens": 200, "output_tokens": 100, "cached_input_tokens": 0, '
        '"cost_usd": 0.02, "timestamp": "2026-07-10T11:00:00+00:00", "duration_ms": 1200}\n'
        "not-json\n",
        encoding="utf-8",
    )

    res = await client.get("/api/v1/ops/summary", headers={"X-Ops-Token": TOKEN})
    assert res.status_code == 200
    body = res.json()

    assert body["total_files"] == 0
    agent_types = {row["agent_type"] for row in body["by_agent_type"]}
    assert agent_types == {"answer", "search"}
    assert len(body["recent_calls"]) == 2
    total_cost = sum(row["cost_usd"] for row in body["by_agent_type"])
    assert pytest.approx(total_cost, rel=1e-6) == 0.03
