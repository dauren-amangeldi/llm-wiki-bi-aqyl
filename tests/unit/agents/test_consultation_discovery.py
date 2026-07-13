"""Unit tests for the AI-advisor discovery step (question generation)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_wiki.agents.consultation import DiscoveryResult, run_discovery

pytestmark = pytest.mark.asyncio


def _llm_returning(payload: dict) -> AsyncMock:
    """Build an LLMClient double whose ``complete()`` returns *payload* as JSON text."""
    llm = AsyncMock()
    usage = MagicMock(cost_usd=0.0)
    llm.complete = AsyncMock(return_value=(json.dumps(payload), usage))
    return llm


async def test_run_discovery_returns_questions_when_context_insufficient() -> None:
    llm = _llm_returning(
        {
            "decision_type": "initiative_scaling",
            "sufficient_context": False,
            "questions": [
                {
                    "id": "q1",
                    "text": "Какой результат важнее в ближайшем горизонте?",
                    "why_it_matters": "Определит приоритет между скоростью и эффектом.",
                    "options": ["Скорость запуска", "Финансовый эффект"],
                    "allow_custom": True,
                    "required": False,
                }
            ],
        }
    )
    chunk_store = MagicMock()
    chunk_store.query = MagicMock(return_value=[])

    result = await run_discovery(
        llm, chunk_store, query="Стоит ли масштабировать пилот?", role="employee", language="ru"
    )

    assert isinstance(result, DiscoveryResult)
    assert result.sufficient_context is False
    assert result.decision_type == "initiative_scaling"
    assert len(result.questions) == 1
    assert result.questions[0].id == "q1"


async def test_run_discovery_caps_questions_at_five() -> None:
    llm = _llm_returning(
        {
            "decision_type": "cost_optimization",
            "sufficient_context": False,
            "questions": [
                {
                    "id": f"q{i}",
                    "text": f"Вопрос {i}",
                    "why_it_matters": "",
                    "options": [],
                    "allow_custom": True,
                    "required": False,
                }
                for i in range(8)
            ],
        }
    )
    chunk_store = MagicMock()
    chunk_store.query = MagicMock(return_value=[])

    result = await run_discovery(llm, chunk_store, query="Как сократить затраты?", role="employee", language="ru")

    assert len(result.questions) == 5


async def test_run_discovery_returns_no_questions_when_context_sufficient() -> None:
    llm = _llm_returning({"decision_type": "market_entry", "sufficient_context": True, "questions": []})
    chunk_store = MagicMock()
    chunk_store.query = MagicMock(return_value=[])

    result = await run_discovery(llm, chunk_store, query="Выходить ли на рынок Алматы?", role="employee", language="ru")

    assert result.sufficient_context is True
    assert result.questions == []
