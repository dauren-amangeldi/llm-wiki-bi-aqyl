"""Unit tests for the Search Agent (LW-6)."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_wiki.agents.search import SearchAgent, SearchResult
from llm_wiki.llm.client import LLMClient


def _mock_llm(response_json: object) -> LLMClient:
    """Build a mock LLMClient that returns *response_json* as a JSON string."""
    mock = MagicMock(spec=LLMClient)
    mock.load_prompt.return_value = "formatted prompt"
    mock.complete = AsyncMock(
        return_value=(json.dumps(response_json), MagicMock())
    )
    return mock  # type: ignore[return-value]


async def test_search_agent_finds_relevant_pages() -> None:
    """SearchAgent.run returns results whose score meets the threshold."""
    llm = _mock_llm([
        {"slug": "transformers", "title": "Transformers", "relevance_score": 0.87, "reasoning": "..."},
        {"slug": "bert", "title": "BERT", "relevance_score": 0.72, "reasoning": "..."},
    ])
    agent = SearchAgent(llm)
    results = await agent.run("some text about transformers", ["Transformers", "BERT"], file_id="t1")

    assert len(results) == 2
    assert results[0].slug == "transformers"
    assert results[0].relevance_score == pytest.approx(0.87)
    assert isinstance(results[0], SearchResult)


async def test_search_agent_returns_empty_for_new_topic() -> None:
    """SearchAgent.run returns [] when all scores are below 0.3."""
    llm = _mock_llm([
        {"slug": "transformers", "title": "Transformers", "relevance_score": 0.1},
        {"slug": "bert", "title": "BERT", "relevance_score": 0.05},
    ])
    agent = SearchAgent(llm)
    results = await agent.run("something completely unrelated", ["Transformers", "BERT"])

    assert results == []


async def test_search_agent_empty_index() -> None:
    """SearchAgent.run returns [] immediately when index_headings is empty."""
    llm = MagicMock(spec=LLMClient)
    agent = SearchAgent(llm)

    results = await agent.run("some text", [], file_id="f1")

    assert results == []
    llm.complete.assert_not_called()  # type: ignore[attr-defined]


async def test_search_agent_llm_error_raises() -> None:
    """SearchAgent.run propagates exceptions from the LLM so Celery can retry."""
    mock = MagicMock(spec=LLMClient)
    mock.load_prompt.return_value = "prompt"
    mock.complete = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
    agent = SearchAgent(mock)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="LLM unavailable"):
        await agent.run("text", ["SomePage"])


async def test_search_agent_sorted_by_score() -> None:
    """SearchAgent returns results sorted by descending relevance_score."""
    llm = _mock_llm([
        {"slug": "b-page", "title": "B Page", "relevance_score": 0.5},
        {"slug": "a-page", "title": "A Page", "relevance_score": 0.9},
        {"slug": "c-page", "title": "C Page", "relevance_score": 0.7},
    ])
    agent = SearchAgent(llm)
    results = await agent.run("text", ["A Page", "B Page", "C Page"])

    scores = [r.relevance_score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0].slug == "a-page"


async def test_search_agent_respects_max_results() -> None:
    """SearchAgent caps output at MAX_RESULTS entries."""
    items = [
        {"slug": f"page-{i}", "title": f"Page {i}", "relevance_score": 0.9 - i * 0.01}
        for i in range(20)
    ]
    llm = _mock_llm(items)
    agent = SearchAgent(llm)
    results = await agent.run("text", [f"Page {i}" for i in range(20)])

    assert len(results) <= SearchAgent.MAX_RESULTS


async def test_search_agent_invalid_json_raises() -> None:
    """SearchAgent raises ValueError when the LLM returns non-JSON."""
    mock = MagicMock(spec=LLMClient)
    mock.load_prompt.return_value = "prompt"
    mock.complete = AsyncMock(return_value=("not valid json at all", MagicMock()))
    agent = SearchAgent(mock)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="invalid JSON"):
        await agent.run("text", ["SomePage"])


async def test_search_agent_non_list_json_raises() -> None:
    """SearchAgent raises ValueError when the LLM returns a JSON object, not array."""
    mock = MagicMock(spec=LLMClient)
    mock.load_prompt.return_value = "prompt"
    mock.complete = AsyncMock(return_value=('{"slug": "x"}', MagicMock()))
    agent = SearchAgent(mock)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="expected a JSON array"):
        await agent.run("text", ["SomePage"])
