"""Unit tests for building the understanding snapshot from Q&A answers."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

from llm_wiki.agents.consultation import build_snapshot
from llm_wiki.api.schemas import ClarificationQuestion, QuestionAnswer

SNAPSHOT_JSON = {
    "decision": "Масштабировать ли пилот",
    "desired_outcome": "Сократить срок решений",
    "horizon": "Текущий год",
    "constraints": ["Команда ограничена"],
    "stakeholders": ["ИТ"],
    "success_criteria": ["Измеримое сокращение"],
    "assumptions": ["Пилот репрезентативен"],
}


async def test_build_snapshot_uses_llm_with_query_and_answers() -> None:
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=(json.dumps(SNAPSHOT_JSON), None))
    questions = [ClarificationQuestion(id="q1", text="Что важнее?", options=["Скорость", "Эффект"])]
    answers = [QuestionAnswer(question_id="q1", answer="Скорость", skipped=False)]

    snapshot = await build_snapshot(llm, query="Стоит ли масштабировать пилот?", questions=questions, answers=answers)

    assert snapshot.decision == "Масштабировать ли пилот"
    assert snapshot.constraints == ["Команда ограничена"]
    llm.complete.assert_awaited_once()


async def test_build_snapshot_degrades_gracefully_on_malformed_json() -> None:
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=("not json at all", None))
    questions: list[ClarificationQuestion] = []
    answers: list[QuestionAnswer] = []

    snapshot = await build_snapshot(llm, query="Стоит ли масштабировать пилот?", questions=questions, answers=answers)

    assert snapshot.decision == "Стоит ли масштабировать пилот?"
    assert snapshot.constraints == []
    assert snapshot.assumptions == []
