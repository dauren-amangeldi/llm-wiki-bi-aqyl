"""Unit tests for TwinsAgent.run_position_round."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_wiki.agents.twins import PositionResult, TwinPersonaData, TwinsAgent
from llm_wiki.llm.client import LLMClient


def _mock_llm(payloads_by_persona: dict[str, dict]) -> LLMClient:
    mock = MagicMock(spec=LLMClient)
    mock.load_prompt.return_value = "prompt"

    async def _complete(*, system: str, **_kwargs: object) -> tuple[str, MagicMock]:
        # system prompt starts with "Ты — <Имя>," — use it to pick the right payload
        for persona_id, payload in payloads_by_persona.items():
            if persona_id in system:
                usage = MagicMock()
                usage.cost_usd = 0.001
                return json.dumps(payload), usage
        raise AssertionError(f"no mock payload matched system prompt: {system}")

    mock.complete = AsyncMock(side_effect=_complete)
    return mock  # type: ignore[return-value]


def _persona(persona_id: str) -> TwinPersonaData:
    return TwinPersonaData(
        id=persona_id,
        lens="test lens",
        system_prompt=f"Ты — {persona_id}, тестовая персона.",
        domain_weights={"tech": 0.5, "real_estate": 0.5, "finance": 0.5},
    )


@pytest.mark.asyncio
async def test_run_position_round_returns_one_result_per_persona() -> None:
    llm = _mock_llm(
        {
            "musk": {"reframing": "r1", "text": "t1", "cite": "c1"},
            "zell": {"reframing": "r2", "text": "t2", "cite": "c2"},
        }
    )
    agent = TwinsAgent(llm)

    results = await agent.run_position_round(
        [_persona("musk"), _persona("zell")], case_context="ctx", language="ru"
    )

    assert len(results) == 2
    by_id = {r.persona_id: r for r in results}
    assert by_id["musk"] == PositionResult(persona_id="musk", reframing="r1", text="t1", cite="c1")
    assert by_id["zell"] == PositionResult(persona_id="zell", reframing="r2", text="t2", cite="c2")
