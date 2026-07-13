"""Integration tests for the Twins council SSE endpoint (BI-AQYL-TWINS)."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

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


async def test_twin_personas_endpoint_returns_seeded_roster(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/twin/personas")

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["personas"]) == 11
    assert len(body["presets"]) == 4


async def test_twin_suggest_returns_null_without_similar_cases(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/twin/suggest", params={"case_id": "case-unknown"})

    assert resp.status_code == 200
    assert resp.json() == {"suggestion": None}


async def test_twin_sessions_outcome_roundtrip(client: AsyncClient, db_engine) -> None:
    from llm_wiki.storage.metadata import create_twin_session

    session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
        bind=db_engine, expire_on_commit=False, autoflush=False
    )
    async with session_factory() as s:
        ts = await create_twin_session(
            s, case_id="case-1", persona_ids=["musk"], created_by="u1"
        )

    listed = await client.get("/api/v1/twin/sessions", params={"case_id": "case-1"})
    assert listed.status_code == 200
    assert listed.json()[0]["id"] == ts.id
    assert listed.json()[0]["outcome"] == ""

    resp = await client.patch(
        f"/api/v1/twin/sessions/{ts.id}/outcome",
        json={"outcome": "confirmed", "note": "риск реализовался, как и предупреждали"},
    )
    assert resp.status_code == 200

    listed = await client.get("/api/v1/twin/sessions", params={"case_id": "case-1"})
    assert listed.json()[0]["outcome"] == "confirmed"
    assert "риск" in listed.json()[0]["outcome_note"]


async def test_twin_session_outcome_404_for_unknown_session(client: AsyncClient) -> None:
    resp = await client.patch(
        "/api/v1/twin/sessions/nope/outcome", json={"outcome": "refuted"}
    )
    assert resp.status_code == 404


def _mock_chat_agent() -> MagicMock:
    from llm_wiki.agents.twins import ChatReplyResult

    agent = MagicMock()
    agent.route_message = AsyncMock(return_value=["musk"])
    agent.respond_as_persona = AsyncMock(
        return_value=ChatReplyResult(persona_id="musk", text="Ответ.", cite="Смета")
    )
    return agent


async def test_twin_chat_streams_typing_message_and_done(client: AsyncClient) -> None:
    mock_agent = _mock_chat_agent()

    with patch("llm_wiki.api.v1.twins.TwinsAgent", return_value=mock_agent), patch(
        "llm_wiki.api.v1.twins.load_case_context", return_value="ctx"
    ), patch("llm_wiki.llm.client.LLMClient") as mock_llm_cls:
        mock_llm = MagicMock()
        mock_llm.aclose = AsyncMock()
        mock_llm_cls.return_value = mock_llm

        resp = await client.post(
            "/api/v1/twin/chat",
            json={"case_id": "case-1", "persona_ids": ["musk"], "message": "Привет", "language": "ru"},
        )

    assert resp.status_code == 200
    body = resp.text
    assert '"event": "typing"' in body
    assert '"persona_id": "musk"' in body
    assert '"text": "\\u041e\\u0442\\u0432\\u0435\\u0442."' in body or "Ответ." in body
    assert '"done": true' in body
    mock_agent.route_message.assert_awaited_once()
    mock_agent.respond_as_persona.assert_awaited_once()


async def test_twin_chat_continues_an_existing_session(client: AsyncClient) -> None:
    mock_agent = _mock_chat_agent()

    with patch("llm_wiki.api.v1.twins.TwinsAgent", return_value=mock_agent), patch(
        "llm_wiki.api.v1.twins.load_case_context", return_value="ctx"
    ), patch("llm_wiki.llm.client.LLMClient") as mock_llm_cls:
        mock_llm = MagicMock()
        mock_llm.aclose = AsyncMock()
        mock_llm_cls.return_value = mock_llm

        # First message creates a session.
        first = await client.post(
            "/api/v1/twin/chat",
            json={"case_id": "case-1", "persona_ids": ["musk"], "message": "Привет", "language": "ru"},
        )
        session_id = json.loads(first.text.strip().splitlines()[-1].removeprefix("data: "))["session_id"]

        # Second message reuses it — route_message/respond_as_persona called again.
        second = await client.post(
            "/api/v1/twin/chat",
            json={
                "session_id": session_id, "case_id": "case-1", "persona_ids": ["musk"],
                "message": "А если иначе?", "language": "ru",
            },
        )

    assert second.status_code == 200
    assert '"done": true' in second.text
    assert mock_agent.route_message.await_count == 2


async def test_twin_chat_second_responder_sees_first_responders_reply(
    client: AsyncClient,
) -> None:
    """The whole point of sequential-with-visibility: persona 2 must see persona 1's
    reply in its `chat_transcript` argument, not just the pre-turn transcript."""
    from llm_wiki.agents.twins import ChatReplyResult

    agent = MagicMock()
    agent.route_message = AsyncMock(return_value=["musk", "zell"])

    async def _respond_as_persona(persona, case_context, chat_transcript, language):
        if persona.id == "musk":
            return ChatReplyResult(persona_id="musk", text="Первый ответ Маска.", cite="")
        return ChatReplyResult(persona_id="zell", text="Ответ Зелла.", cite="")

    agent.respond_as_persona = AsyncMock(side_effect=_respond_as_persona)

    with patch("llm_wiki.api.v1.twins.TwinsAgent", return_value=agent), patch(
        "llm_wiki.api.v1.twins.load_case_context", return_value="ctx"
    ), patch("llm_wiki.llm.client.LLMClient") as mock_llm_cls:
        mock_llm = MagicMock()
        mock_llm.aclose = AsyncMock()
        mock_llm_cls.return_value = mock_llm

        resp = await client.post(
            "/api/v1/twin/chat",
            json={
                "case_id": "case-1", "persona_ids": ["musk", "zell"],
                "message": "Привет", "language": "ru",
            },
        )

    assert resp.status_code == 200
    assert agent.respond_as_persona.await_count == 2
    second_call_kwargs = agent.respond_as_persona.call_args_list[1]
    second_transcript = second_call_kwargs.args[2] if len(second_call_kwargs.args) > 2 else second_call_kwargs.kwargs["chat_transcript"]
    assert "Первый ответ Маска." in second_transcript


async def test_twin_chat_transcript_accumulates_across_turns(client: AsyncClient) -> None:
    """A second turn on the same session must see the first turn's user message
    and persona reply in its transcript — this is what makes conversations
    coherent across multiple `/twin/chat` calls."""
    from llm_wiki.agents.twins import ChatReplyResult

    agent = MagicMock()
    agent.route_message = AsyncMock(return_value=["musk"])
    agent.respond_as_persona = AsyncMock(
        return_value=ChatReplyResult(persona_id="musk", text="Ответ из первого хода.", cite="")
    )

    with patch("llm_wiki.api.v1.twins.TwinsAgent", return_value=agent), patch(
        "llm_wiki.api.v1.twins.load_case_context", return_value="ctx"
    ), patch("llm_wiki.llm.client.LLMClient") as mock_llm_cls:
        mock_llm = MagicMock()
        mock_llm.aclose = AsyncMock()
        mock_llm_cls.return_value = mock_llm

        first = await client.post(
            "/api/v1/twin/chat",
            json={
                "case_id": "case-1", "persona_ids": ["musk"],
                "message": "Первое сообщение пользователя", "language": "ru",
            },
        )
        session_id = json.loads(first.text.strip().splitlines()[-1].removeprefix("data: "))["session_id"]

        second = await client.post(
            "/api/v1/twin/chat",
            json={
                "session_id": session_id, "case_id": "case-1", "persona_ids": ["musk"],
                "message": "Второе сообщение", "language": "ru",
            },
        )

    assert second.status_code == 200
    second_turn_transcript = agent.route_message.call_args_list[1].args[1]
    assert "Первое сообщение пользователя" in second_turn_transcript
    assert "Ответ из первого хода." in second_turn_transcript


async def test_twin_chat_rejects_case_id_mismatch_with_existing_session(
    client: AsyncClient, db_engine
) -> None:
    from llm_wiki.storage.metadata import create_twin_session

    session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
        bind=db_engine, expire_on_commit=False, autoflush=False
    )
    async with session_factory() as s:
        ts = await create_twin_session(s, case_id="case-1", persona_ids=["musk"], created_by="u1")

    resp = await client.post(
        "/api/v1/twin/chat",
        json={
            "session_id": ts.id, "case_id": "case-2", "persona_ids": ["musk"],
            "message": "Привет", "language": "ru",
        },
    )
    assert resp.status_code == 400


async def test_twin_chat_rejects_more_than_three_personas(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/twin/chat",
        json={"case_id": "case-1", "persona_ids": ["musk", "zell", "bren", "hines"], "message": "hi"},
    )
    assert resp.status_code == 422


async def test_twin_chat_dedupes_repeated_persona_ids(client: AsyncClient) -> None:
    """A client sending the same persona id twice must get one reply, not two."""
    from llm_wiki.agents.twins import ChatReplyResult

    agent = MagicMock()
    agent.route_message = AsyncMock(return_value=["musk"])
    agent.respond_as_persona = AsyncMock(
        return_value=ChatReplyResult(persona_id="musk", text="Ответ.", cite="")
    )

    with patch("llm_wiki.api.v1.twins.TwinsAgent", return_value=agent), patch(
        "llm_wiki.api.v1.twins.load_case_context", return_value="ctx"
    ), patch("llm_wiki.llm.client.LLMClient") as mock_llm_cls:
        mock_llm = MagicMock()
        mock_llm.aclose = AsyncMock()
        mock_llm_cls.return_value = mock_llm

        resp = await client.post(
            "/api/v1/twin/chat",
            json={
                "case_id": "case-1", "persona_ids": ["musk", "musk"],
                "message": "Привет", "language": "ru",
            },
        )

    assert resp.status_code == 200
    assert agent.respond_as_persona.await_count == 1


async def test_summarize_session_returns_verdict_and_persists_it(
    client: AsyncClient, db_engine
) -> None:
    from llm_wiki.agents.twins import VerdictResult
    from llm_wiki.storage.metadata import append_twin_message, create_twin_session, get_twin_session_messages

    session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
        bind=db_engine, expire_on_commit=False, autoflush=False
    )
    async with session_factory() as s:
        ts = await create_twin_session(s, case_id="case-1", persona_ids=["musk"], created_by="u1")
        await append_twin_message(
            s, session_id=ts.id, role="user", persona_id=None, seq=0, content={"text": "Вопрос"}
        )

    mock_agent = MagicMock()
    mock_agent.run_chat_verdict = AsyncMock(
        return_value=VerdictResult(
            questions=[], consensus="ok", disagreement="none", next_step="ship it",
            domain_distribution={"tech": 1.0, "real_estate": 0.0, "finance": 0.0},
            decisive_voice="musk", consensus_reached_early=False, is_close_split=False,
        )
    )

    with patch("llm_wiki.api.v1.twins.TwinsAgent", return_value=mock_agent), patch(
        "llm_wiki.api.v1.twins.load_case_context", return_value="ctx"
    ), patch("llm_wiki.llm.client.LLMClient") as mock_llm_cls:
        mock_llm = MagicMock()
        mock_llm.aclose = AsyncMock()
        mock_llm_cls.return_value = mock_llm

        resp = await client.post(f"/api/v1/twin/sessions/{ts.id}/summarize")

    assert resp.status_code == 200
    body = resp.json()
    assert body["consensus"] == "ok"
    assert body["decisive_voice"] == "musk"

    async with session_factory() as s:
        messages = await get_twin_session_messages(s, ts.id)
    assert any(m.role == "verdict" for m in messages)


async def test_summarize_session_404_for_unknown_session(client: AsyncClient) -> None:
    resp = await client.post("/api/v1/twin/sessions/nope/summarize")
    assert resp.status_code == 404


async def test_summarize_session_400_when_chat_is_empty(client: AsyncClient, db_engine) -> None:
    from llm_wiki.storage.metadata import create_twin_session

    session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
        bind=db_engine, expire_on_commit=False, autoflush=False
    )
    async with session_factory() as s:
        ts = await create_twin_session(s, case_id="case-1", persona_ids=["musk"], created_by="u1")

    resp = await client.post(f"/api/v1/twin/sessions/{ts.id}/summarize")
    assert resp.status_code == 400
