"""Unit tests for TwinsAgent's domain-weighted decisive-voice vote and chat verdict."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_wiki.agents.twins import TwinPersonaData, TwinsAgent, _compute_decisive_voice
from llm_wiki.llm.client import LLMClient


def test_compute_decisive_voice_picks_highest_weighted_score() -> None:
    weights = {
        "musk": {"tech": 0.9, "real_estate": 0.2, "finance": 0.3},
        "zell": {"tech": 0.1, "real_estate": 0.7, "finance": 0.9},
    }
    decisive_id, is_close = _compute_decisive_voice(
        ["musk", "zell"], weights, {"tech": 0.1, "real_estate": 0.1, "finance": 0.8}
    )
    assert decisive_id == "zell"
    assert is_close is False


def test_compute_decisive_voice_flags_close_split_under_threshold() -> None:
    weights = {
        "musk": {"tech": 0.6, "real_estate": 0.5, "finance": 0.5},
        "zell": {"tech": 0.5, "real_estate": 0.5, "finance": 0.55},
    }
    _decisive_id, is_close = _compute_decisive_voice(
        ["musk", "zell"], weights, {"tech": 0.34, "real_estate": 0.33, "finance": 0.33}
    )
    assert is_close is True


def test_compute_decisive_voice_not_close_split_over_threshold() -> None:
    weights = {
        "musk": {"tech": 1.0, "real_estate": 0.0, "finance": 0.0},
        "zell": {"tech": 0.0, "real_estate": 1.0, "finance": 1.0},
    }
    _decisive_id, is_close = _compute_decisive_voice(
        ["musk", "zell"], weights, {"tech": 0.34, "real_estate": 0.33, "finance": 0.33}
    )
    assert is_close is False


def _persona(persona_id: str) -> TwinPersonaData:
    return TwinPersonaData(
        id=persona_id, lens="test lens", system_prompt=f"Ты — {persona_id}, тестовая персона.",
        domain_weights={"tech": 0.9, "real_estate": 0.1, "finance": 0.1} if persona_id == "musk"
        else {"tech": 0.1, "real_estate": 0.1, "finance": 0.9},
    )


@pytest.mark.asyncio
async def test_run_chat_verdict_summarizes_free_form_transcript() -> None:
    mock = MagicMock(spec=LLMClient)
    mock.load_prompt.return_value = "prompt"
    usage = MagicMock()
    mock.complete = AsyncMock(
        return_value=(
            json.dumps({
                "questions": [{"text": "q1", "persona_id": "musk"}],
                "consensus": "agree", "disagreement": "timing", "next_step": "stress-test",
                "domain_distribution": {"tech": 0.1, "real_estate": 0.1, "finance": 0.8},
            }),
            usage,
        )
    )
    agent = TwinsAgent(mock)

    verdict = await agent.run_chat_verdict(
        [_persona("musk"), _persona("zell")], case_context="ctx",
        chat_transcript="Пользователь: вопрос\nElon Musk: ответ", language="ru",
    )

    assert verdict.decisive_voice == "zell"  # finance-weighted domain won
    assert verdict.consensus_reached_early is False
    assert verdict.domain_distribution == {"tech": 0.1, "real_estate": 0.1, "finance": 0.8}
