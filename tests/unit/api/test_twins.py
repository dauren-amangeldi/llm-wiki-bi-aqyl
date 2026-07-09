"""Integration tests for the Twins council SSE endpoint (BI-AQYL-TWINS)."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llm_wiki.agents.twins import CrossExamResult, PositionResult, VerdictResult
from llm_wiki.api.deps import get_db
from llm_wiki.main import app
from llm_wiki.storage.metadata import seed_twin_personas


# `settings.database_url` (what `get_db` uses by default) is not necessarily the
# same connection the `db_engine` fixture builds. Every other API-level test that
# needs real DB state overrides `get_db` to point at the per-test engine — see
# `tests/unit/api/test_cases.py` for the established pattern. Twins needs this too
# because the roster endpoint reads real seeded rows and the council endpoint
# writes a real session/message.
@pytest_asyncio.fixture
async def client(db_engine) -> AsyncGenerator[AsyncClient, None]:
    session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
        bind=db_engine, expire_on_commit=False, autoflush=False
    )

    async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db

    # Seed personas into the SAME per-test engine the override above uses —
    # app startup's lifespan seeding never runs against this throwaway schema.
    async with session_factory() as seed_session:
        await seed_twin_personas(seed_session)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"  # type: ignore[arg-type]
    ) as ac:
        yield ac

    app.dependency_overrides.pop(get_db, None)


def _mock_agent() -> MagicMock:
    agent = MagicMock()
    agent.run_position_round = AsyncMock(
        return_value=[PositionResult(persona_id="musk", reframing="r", text="t", cite="c")]
    )
    agent.run_cross_exam_round = AsyncMock(return_value=[])
    agent.run_verdict_round = AsyncMock(
        return_value=VerdictResult(
            questions=[], consensus="ok", disagreement="none", next_step="ship it",
            domain_distribution={"tech": 1.0, "real_estate": 0.0, "finance": 0.0},
            decisive_voice="musk", consensus_reached_early=False, is_close_split=False,
        )
    )
    return agent


async def test_twin_personas_endpoint_returns_seeded_roster(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/twin/personas")

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["personas"]) == 11
    assert len(body["presets"]) == 4


async def test_twin_council_sse_stream_returns_done_event(client: AsyncClient) -> None:
    mock_agent = _mock_agent()

    with patch("llm_wiki.api.v1.twins.TwinsAgent", return_value=mock_agent), patch(
        "llm_wiki.api.v1.twins.load_case_context", return_value="ctx"
    ), patch("llm_wiki.llm.client.LLMClient") as mock_llm_cls:
        mock_llm = MagicMock()
        mock_llm.aclose = AsyncMock()
        mock_llm_cls.return_value = mock_llm

        resp = await client.post(
            "/api/v1/twin/council",
            params={"stream": "true"},
            json={"case_id": "case-does-not-exist", "persona_ids": ["musk"], "language": "ru"},
        )

    assert resp.status_code == 200
    body = resp.text
    assert '"round": "position"' in body
    assert '"round": "verdict"' in body
    assert '"is_close_split": false' in body
    assert '"done": true' in body
    mock_agent.run_position_round.assert_awaited_once()
    mock_agent.run_verdict_round.assert_awaited_once()


async def test_twin_council_rejects_more_than_three_personas(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/twin/council",
        json={"case_id": "case-1", "persona_ids": ["musk", "zell", "bren", "hines"], "language": "ru"},
    )

    assert resp.status_code == 422
