"""Unit tests for TwinsAgent's chat primitives: router, responder, transcript builder."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_wiki.agents.twins import ChatReplyResult, TwinPersonaData, TwinsAgent, build_chat_transcript
from llm_wiki.llm.client import LLMClient


def _persona(persona_id: str, lens: str = "test lens") -> TwinPersonaData:
    return TwinPersonaData(
        id=persona_id, lens=lens, system_prompt=f"Ты — {persona_id}.",
        domain_weights={"tech": 0.5, "real_estate": 0.3, "finance": 0.2},
    )


class _FakeMessage:
    def __init__(self, role: str, persona_id: str | None, text: str) -> None:
        self.role = role
        self.persona_id = persona_id
        self.content = {"text": text}


def test_build_chat_transcript_renders_speakers_by_real_name() -> None:
    messages = [
        _FakeMessage("user", None, "Себестоимость выросла на 12%"),
        _FakeMessage("persona", "musk", "Разбери смету по статьям."),
    ]
    transcript = build_chat_transcript(messages, {"musk": "Elon Musk"})

    assert "Пользователь: Себестоимость выросла на 12%" in transcript
    assert "Elon Musk: Разбери смету по статьям." in transcript


def test_build_chat_transcript_labels_verdict_rows() -> None:
    messages = [_FakeMessage("verdict", None, "")]
    transcript = build_chat_transcript(messages, {})
    assert transcript.startswith("Итог:")


@pytest.mark.asyncio
async def test_route_message_returns_only_known_persona_ids() -> None:
    mock = MagicMock(spec=LLMClient)
    mock.load_prompt.return_value = "prompt"
    usage = MagicMock()
    mock.complete = AsyncMock(
        return_value=(json.dumps({"responders": ["musk", "ghost", "zell"]}), usage)
    )
    agent = TwinsAgent(mock)

    responders = await agent.route_message(
        [_persona("musk"), _persona("zell")], chat_transcript="...", language="ru",
    )

    assert responders == ["musk", "zell"]  # "ghost" filtered out — not in the roster


@pytest.mark.asyncio
async def test_route_message_can_return_empty_list() -> None:
    mock = MagicMock(spec=LLMClient)
    mock.load_prompt.return_value = "prompt"
    usage = MagicMock()
    mock.complete = AsyncMock(return_value=(json.dumps({"responders": []}), usage))
    agent = TwinsAgent(mock)

    responders = await agent.route_message([_persona("musk")], chat_transcript="...", language="ru")

    assert responders == []


@pytest.mark.asyncio
async def test_respond_as_persona_returns_text_and_cite() -> None:
    mock = MagicMock(spec=LLMClient)
    mock.load_prompt.return_value = "prompt"
    usage = MagicMock()
    mock.complete = AsyncMock(
        return_value=(json.dumps({"text": "Ответ.", "cite": "Смета · разд. 3"}), usage)
    )
    agent = TwinsAgent(mock)

    reply = await agent.respond_as_persona(
        _persona("musk"), case_context="ctx", chat_transcript="...", language="ru",
    )

    assert isinstance(reply, ChatReplyResult)
    assert reply.persona_id == "musk"
    assert reply.text == "Ответ."
    assert reply.cite == "Смета · разд. 3"
