"""Unit tests for TwinsAgent.run_cross_exam_round and the disagreement-novelty gate."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_wiki.agents.twins import (
    CrossExamResult,
    PositionResult,
    TwinPersonaData,
    TwinsAgent,
    _is_novel_disagreement,
)
from llm_wiki.llm.client import LLMClient


def test_is_novel_disagreement_true_for_distinct_text() -> None:
    assert _is_novel_disagreement("Совсем другая мысль про риск", ["Позиция про себестоимость"])


def test_is_novel_disagreement_false_for_near_duplicate() -> None:
    prior = "Себестоимость метра слишком высокая из-за ручного труда на площадке."
    candidate = "себестоимость метра слишком высокая из-за ручного труда на площадке"
    assert not _is_novel_disagreement(candidate, [prior])


def _persona(persona_id: str) -> TwinPersonaData:
    return TwinPersonaData(
        id=persona_id,
        lens="test lens",
        system_prompt=f"Ты — {persona_id}, тестовая персона.",
        domain_weights={"tech": 0.5, "real_estate": 0.5, "finance": 0.5},
    )


def _mock_llm(payloads_by_persona: dict[str, list[dict]]) -> LLMClient:
    """Create a mock LLMClient that returns payloads in order per persona."""
    mock = MagicMock(spec=LLMClient)
    mock.load_prompt.return_value = "prompt"

    # Track call counts per persona
    call_counts: dict[str, int] = {}

    async def _complete(*, system: str, **_kwargs: object) -> tuple[str, MagicMock]:
        # system prompt starts with "Ты — <Имя>," — use it to pick the right payload list
        for persona_id, payloads in payloads_by_persona.items():
            if persona_id in system:
                idx = call_counts.get(persona_id, 0)
                call_counts[persona_id] = idx + 1
                if idx >= len(payloads):
                    raise AssertionError(
                        f"persona {persona_id} exhausted payloads: "
                        f"tried to call {idx + 1}, but only {len(payloads)} available"
                    )
                usage = MagicMock()
                usage.cost_usd = 0.001
                return json.dumps(payloads[idx]), usage
        raise AssertionError(f"no mock payload matched system prompt: {system}")

    mock.complete = AsyncMock(side_effect=_complete)
    return mock  # type: ignore[return-value]


@pytest.mark.asyncio
async def test_cross_exam_accepts_novel_disagreement_on_first_try() -> None:
    mock = _mock_llm(
        {
            "musk": [
                {"disagreement": "Новая мысль про риск", "text": "reply", "cite": "c"}
            ]
        }
    )
    agent = TwinsAgent(mock)
    positions = [
        PositionResult(
            persona_id="musk",
            reframing="r",
            text="Позиция про себестоимость",
            cite="c",
        )
    ]

    results = await agent.run_cross_exam_round(
        [_persona("musk")], case_context="ctx", positions=positions, language="ru"
    )

    assert results == [
        CrossExamResult(
            persona_id="musk",
            disagreement="Новая мысль про риск",
            disagreement_forced=False,
            text="reply",
            cite="c",
        )
    ]
    assert mock.complete.await_count == 1


@pytest.mark.asyncio
async def test_cross_exam_marks_forced_when_model_repeats_own_position_twice() -> None:
    repeated = "Позиция про себестоимость"
    mock = _mock_llm(
        {
            "musk": [
                {"disagreement": repeated, "text": "reply", "cite": "c"},
                {"disagreement": repeated, "text": "reply", "cite": "c"},
            ]
        }
    )
    agent = TwinsAgent(mock)
    positions = [
        PositionResult(persona_id="musk", reframing="r", text=repeated, cite="c")
    ]

    results = await agent.run_cross_exam_round(
        [_persona("musk")], case_context="ctx", positions=positions, language="ru"
    )

    assert results[0].disagreement_forced is True
    assert mock.complete.await_count == 2  # one retry after the first duplicate


@pytest.mark.asyncio
async def test_cross_exam_skips_failed_persona_keeps_others() -> None:
    async def _complete(*, system: str, **_kwargs: object) -> tuple[str, MagicMock]:
        if "musk" in system:
            raise RuntimeError("boom")
        usage = MagicMock()
        usage.cost_usd = 0.001
        return json.dumps(
            {"disagreement": "新しい考え", "text": "t", "cite": "c"}
        ), usage

    mock = MagicMock(spec=LLMClient)
    mock.load_prompt.return_value = "prompt"
    mock.complete = AsyncMock(side_effect=_complete)
    agent = TwinsAgent(mock)

    positions = [
        PositionResult(persona_id="musk", reframing="r", text="t1", cite="c"),
        PositionResult(persona_id="zell", reframing="r", text="t2", cite="c"),
    ]

    results = await agent.run_cross_exam_round(
        [_persona("musk"), _persona("zell")],
        case_context="ctx",
        positions=positions,
        language="ru",
    )

    assert [r.persona_id for r in results] == ["zell"]


@pytest.mark.asyncio
async def test_cross_exam_falls_back_to_first_attempt_when_retry_raises() -> None:
    call_count = 0

    async def _complete(**_kwargs: object) -> tuple[str, MagicMock]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            usage = MagicMock()
            usage.cost_usd = 0.001
            return json.dumps({"disagreement": "Позиция про себестоимость", "text": "reply1", "cite": "c1"}), usage
        raise RuntimeError("network blip")

    mock = MagicMock(spec=LLMClient)
    mock.load_prompt.return_value = "prompt"
    mock.complete = AsyncMock(side_effect=_complete)
    agent = TwinsAgent(mock)
    positions = [PositionResult(persona_id="musk", reframing="r", text="Позиция про себестоимость", cite="c")]

    results = await agent.run_cross_exam_round(
        [_persona("musk")], case_context="ctx", positions=positions, language="ru"
    )

    assert results == [
        CrossExamResult(persona_id="musk", disagreement="Позиция про себестоимость", disagreement_forced=True, text="reply1", cite="c1")
    ]
