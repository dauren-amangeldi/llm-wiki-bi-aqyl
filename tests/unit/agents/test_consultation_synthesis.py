"""Unit tests for the AI-advisor synthesis step (decision brief + internal critique)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_wiki.agents.consultation import run_synthesis
from llm_wiki.api.schemas import UnderstandingSnapshot
from llm_wiki.llm.chunk_store import ChunkHit

pytestmark = pytest.mark.asyncio

DRAFT_JSON = {
    "recommendation": "Ограниченное масштабирование по двум подразделениям",
    "why_now": "Пилот дал сигнал",
    "problem_frame": "Масштабировать сразу или проверить переносимость",
    "key_assumption": "Эффект пилота сохранится",
    "rationale": "Проверяет переносимость без полного обязательства",
    "alternatives": ["Полное масштабирование", "Статус-кво"],
    "risks": ["Другая операционная модель"],
    "first_step": "Выбрать два подразделения",
    "reconsider_if": ["Эффект ниже 20% от пилота"],
    "evidence_strength": "medium",
    "assumptions": ["Пилот репрезентативен"],
    "sources": ["Отчёт по пилоту"],
}

CRITIQUE_JSON = {
    "risks": ["Другая операционная модель", "Сопротивление среднего менеджмента"],
    "evidence_strength": "medium",
    "reconsider_if": ["Эффект ниже 20% от пилота"],
}


def _snapshot() -> UnderstandingSnapshot:
    return UnderstandingSnapshot(
        decision="Масштабировать ли пилот",
        desired_outcome="Сократить срок решений",
        horizon="Текущий год",
        constraints=["Команда ограничена"],
        stakeholders=["ИТ"],
        success_criteria=["Измеримое сокращение"],
        assumptions=["Пилот репрезентативен"],
    )


def _chunk_store_with_hits() -> MagicMock:
    chunk_store = MagicMock()
    chunk_store.query = MagicMock(
        return_value=[
            ChunkHit(
                slug="pilot-report",
                title="Отчёт по пилоту",
                section="",
                chunk_idx=0,
                text="Отчёт по пилоту",
                similarity=0.9,
                file_id="",
            ),
            ChunkHit(
                slug="how-to-sell",
                title="Кейс Как продать / внедрить",
                section="",
                chunk_idx=0,
                text="Кейс Как продать / внедрить",
                similarity=0.8,
                file_id="",
            ),
        ]
    )
    return chunk_store


async def test_run_synthesis_returns_decision_brief() -> None:
    llm = AsyncMock()
    usage = MagicMock(cost_usd=0.0)
    # Первый вызов — синтез рекомендации, второй — внутренняя критика,
    # которая уточняет риски/уровень доказательности без отдельного вывода пользователю.
    llm.complete = AsyncMock(
        side_effect=[
            (json.dumps(DRAFT_JSON), usage),
            (json.dumps(CRITIQUE_JSON), usage),
        ]
    )
    chunk_store = _chunk_store_with_hits()

    brief = await run_synthesis(
        llm, chunk_store, query="Стоит ли масштабировать пилот?", snapshot=_snapshot(), language="ru"
    )

    assert brief.recommendation.startswith("Ограниченное масштабирование")
    # Критический проход дополнил риски — их должно быть больше, чем в первом черновике.
    assert len(brief.risks) == 2
    assert llm.complete.await_count == 2


async def test_run_synthesis_degrades_gracefully_on_malformed_draft_json() -> None:
    """LLM returning prose instead of JSON on the draft call must not crash run_synthesis."""
    llm = AsyncMock()
    usage = MagicMock(cost_usd=0.0)
    llm.complete = AsyncMock(
        return_value=("Извините, не могу сформировать структурированный ответ.", usage)
    )
    chunk_store = _chunk_store_with_hits()

    brief = await run_synthesis(
        llm, chunk_store, query="Стоит ли масштабировать пилот?", snapshot=_snapshot(), language="ru"
    )

    assert brief.evidence_strength == "low"
    assert brief.recommendation


async def test_run_synthesis_keeps_draft_when_critique_json_is_malformed() -> None:
    """A malformed critique reply degrades to the draft brief, not an exception."""
    llm = AsyncMock()
    usage = MagicMock(cost_usd=0.0)
    llm.complete = AsyncMock(
        side_effect=[
            (json.dumps(DRAFT_JSON), usage),
            ("не критика, а извинение", usage),
        ]
    )
    chunk_store = _chunk_store_with_hits()

    brief = await run_synthesis(
        llm, chunk_store, query="Стоит ли масштабировать пилот?", snapshot=_snapshot(), language="ru"
    )

    assert brief.recommendation.startswith("Ограниченное масштабирование")
    assert brief.risks == ["Другая операционная модель"]
    assert llm.complete.await_count == 2
