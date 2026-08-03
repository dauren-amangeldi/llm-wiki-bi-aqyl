"""Unit tests for dynamic advisor clarifying-question generation."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_wiki.agents.advisor_questions import (
    DECISION_TYPE_LABELS,
    QUESTION_SETS,
    generate_questions,
)
from llm_wiki.llm.client import LLMClient


def _mock_llm_returning(payload: dict) -> LLMClient:  # type: ignore[type-arg]
    mock = MagicMock(spec=LLMClient)
    usage = MagicMock()
    usage.cost_usd = 0.0
    mock.complete = AsyncMock(return_value=(json.dumps(payload), usage))
    return mock  # type: ignore[return-value]


def _mock_llm_raising() -> LLMClient:  # type: ignore[type-arg]
    mock = MagicMock(spec=LLMClient)
    mock.complete = AsyncMock(side_effect=RuntimeError("llm down"))
    return mock  # type: ignore[return-value]


@pytest.mark.asyncio
async def test_uses_llm_generated_questions() -> None:
    """Well-formed model output is returned verbatim (tailored, not static)."""
    payload = {
        "decision_type": "partnership",
        "questions": [
            {"id": "proved", "text": "Что продукт доказал?", "options": ["A", "B"], "multi": False},
            {"id": "wants", "text": "Что хочет партнёр?", "options": ["X", "Y", "Z"], "multi": True},
        ],
    }
    llm = _mock_llm_returning(payload)
    result = await generate_questions(llm, "Партнёр хочет долю", "ru")

    assert result["decision_type"] == "partnership"
    assert [q["id"] for q in result["questions"]] == ["proved", "wants"]
    assert result["questions"][1]["multi"] is True
    # These are NOT the static partnership set.
    assert result["questions"] != QUESTION_SETS["partnership"]


@pytest.mark.asyncio
async def test_normalizes_caps_and_dedupes() -> None:
    """>5 questions capped to 5, <2-option / textless dropped, options capped to 4, ids deduped."""
    payload = {
        "decision_type": "generic",
        "questions": [
            {"id": "dup", "text": "Q1", "options": ["a", "b", "c", "d", "e"], "multi": False},
            {"id": "dup", "text": "Q2", "options": ["a", "b"], "multi": False},
            {"id": "bad_one_option", "text": "Q3", "options": ["only"], "multi": False},
            {"id": "no_text", "text": "", "options": ["a", "b"], "multi": False},
            {"id": "q4", "text": "Q4", "options": ["a", "b"], "multi": False},
            {"id": "q5", "text": "Q5", "options": ["a", "b"], "multi": False},
            {"id": "q6", "text": "Q6", "options": ["a", "b"], "multi": False},
            {"id": "q7", "text": "Q7", "options": ["a", "b"], "multi": False},
        ],
    }
    llm = _mock_llm_returning(payload)
    result = await generate_questions(llm, "situation", "ru")
    qs = result["questions"]

    assert len(qs) == 5  # capped
    assert len(qs[0]["options"]) == 4  # options capped
    ids = [q["id"] for q in qs]
    assert len(ids) == len(set(ids))  # deduped
    texts = [q["text"] for q in qs]
    assert "Q3" not in texts and "" not in texts  # malformed dropped


@pytest.mark.asyncio
async def test_invalid_decision_type_defaults_to_generic() -> None:
    """An unknown decision_type is coerced to 'generic' but LLM questions kept."""
    payload = {
        "decision_type": "banana",
        "questions": [
            {"id": "q1", "text": "Q1", "options": ["a", "b"], "multi": False},
        ],
    }
    llm = _mock_llm_returning(payload)
    result = await generate_questions(llm, "x", "ru")

    assert result["decision_type"] == "generic"
    assert result["decision_type"] in DECISION_TYPE_LABELS
    assert result["questions"][0]["id"] == "q1"


@pytest.mark.asyncio
async def test_falls_back_to_static_on_llm_error() -> None:
    """When the LLM call fails, the curated static set is returned (flow unbroken)."""
    llm = _mock_llm_raising()
    result = await generate_questions(llm, "Партнёр хочет долю", "ru")

    # classify also fails -> 'generic' -> its static set.
    assert result["decision_type"] == "generic"
    assert result["questions"] == QUESTION_SETS["generic"]


@pytest.mark.asyncio
async def test_falls_back_when_all_questions_malformed() -> None:
    """Valid JSON but no usable questions -> fallback rather than an empty set."""
    payload = {
        "decision_type": "partnership",
        "questions": [{"id": "x", "text": "", "options": []}],
    }
    llm = _mock_llm_returning(payload)
    result = await generate_questions(llm, "y", "ru")

    assert len(result["questions"]) >= 1  # never returns an empty set
