"""Unit tests for the Search Agent (LW-6).

Covers:
  - SearchAgent.run() integration (mocked LLM)
  - _parse_search_response() defensive parser for all known response shapes
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_wiki.agents.search import SearchAgent, SearchResult, _parse_search_response
from llm_wiki.llm.client import LLMClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_llm(raw_response: str | object) -> LLMClient:
    """Build a mock LLMClient that returns *raw_response* (stringified if needed)."""
    if not isinstance(raw_response, str):
        raw_response = json.dumps(raw_response)
    mock = MagicMock(spec=LLMClient)
    mock.load_prompt.return_value = "formatted prompt"
    mock.complete = AsyncMock(return_value=(raw_response, MagicMock()))
    return mock  # type: ignore[return-value]


def _candidates(*items: dict) -> str:  # type: ignore[type-arg]
    """Build the canonical `{"candidates": [...]}` JSON string."""
    return json.dumps({"candidates": list(items)})


def _item(slug: str, score: float, title: str = "") -> dict:  # type: ignore[type-arg]
    return {"slug": slug, "title": title or slug.replace("-", " ").title(), "relevance_score": score}


# ===========================================================================
# SearchAgent.run() integration tests
# ===========================================================================


async def test_search_agent_finds_relevant_pages() -> None:
    """SearchAgent.run returns results whose score meets the threshold."""
    llm = _mock_llm(_candidates(
        _item("transformers", 0.87),
        _item("bert", 0.72),
    ))
    agent = SearchAgent(llm)
    results = await agent.run("some text about transformers", ["Transformers", "BERT"], file_id="t1")

    assert len(results) == 2
    assert results[0].slug == "transformers"
    assert results[0].relevance_score == pytest.approx(0.87)
    assert isinstance(results[0], SearchResult)


async def test_search_agent_bare_array_still_accepted() -> None:
    """Bare JSON array (Ollama / legacy) is accepted by the defensive parser."""
    llm = _mock_llm([
        {"slug": "transformers", "title": "Transformers", "relevance_score": 0.87},
        {"slug": "bert", "title": "BERT", "relevance_score": 0.72},
    ])
    agent = SearchAgent(llm)
    results = await agent.run("text", ["Transformers", "BERT"])
    assert len(results) == 2


async def test_search_agent_returns_empty_for_new_topic() -> None:
    """SearchAgent.run returns [] when all scores are below 0.3."""
    llm = _mock_llm(_candidates(
        _item("transformers", 0.1),
        _item("bert", 0.05),
    ))
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
    llm = _mock_llm(_candidates(
        _item("b-page", 0.5),
        _item("a-page", 0.9),
        _item("c-page", 0.7),
    ))
    agent = SearchAgent(llm)
    results = await agent.run("text", ["A Page", "B Page", "C Page"])

    scores = [r.relevance_score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0].slug == "a-page"


async def test_search_agent_respects_max_results() -> None:
    """SearchAgent caps output at MAX_RESULTS entries."""
    items = [_item(f"page-{i}", 0.9 - i * 0.01) for i in range(20)]
    llm = _mock_llm({"candidates": items})
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


async def test_search_agent_unextractable_json_raises() -> None:
    """SearchAgent raises ValueError when the LLM returns a JSON dict with no array."""
    mock = MagicMock(spec=LLMClient)
    mock.load_prompt.return_value = "prompt"
    mock.complete = AsyncMock(return_value=('{"status": "ok"}', MagicMock()))
    agent = SearchAgent(mock)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="cannot extract candidates"):
        await agent.run("text", ["SomePage"])


# ===========================================================================
# _parse_search_response — unit tests for every accepted shape
# ===========================================================================


def test_parse_candidates_wrapper() -> None:
    """{"candidates": [...]} — canonical OpenAI json_object format."""
    raw = json.dumps({"candidates": [{"slug": "x", "relevance_score": 0.8}]})
    items = _parse_search_response(raw)
    assert len(items) == 1
    assert items[0]["slug"] == "x"


def test_parse_results_alias() -> None:
    """{"results": [...]} — common LLM alias for candidates."""
    raw = json.dumps({"results": [{"slug": "y", "relevance_score": 0.7}]})
    items = _parse_search_response(raw)
    assert items[0]["slug"] == "y"


def test_parse_bare_array() -> None:
    """[...] — bare array produced by Ollama and legacy prompts."""
    raw = json.dumps([{"slug": "z", "relevance_score": 0.6}])
    items = _parse_search_response(raw)
    assert items[0]["slug"] == "z"


def test_parse_markdown_fence_candidates() -> None:
    """```json\\n{"candidates": [...]}\\n``` — fenced block from some models."""
    payload = {"candidates": [{"slug": "fenced", "relevance_score": 0.9}]}
    raw = "```json\n" + json.dumps(payload) + "\n```"
    items = _parse_search_response(raw)
    assert items[0]["slug"] == "fenced"


def test_parse_markdown_fence_bare_array() -> None:
    """```json\\n[...]\\n``` — fenced bare array."""
    payload = [{"slug": "fenced-arr", "relevance_score": 0.5}]
    raw = "```\n" + json.dumps(payload) + "\n```"
    items = _parse_search_response(raw)
    assert items[0]["slug"] == "fenced-arr"


def test_parse_single_candidate_flat_object() -> None:
    """{"slug": ..., ...} — single candidate returned as flat dict."""
    raw = json.dumps({"slug": "solo", "title": "Solo", "relevance_score": 0.55})
    items = _parse_search_response(raw)
    assert items[0]["slug"] == "solo"


def test_parse_empty_candidates_list() -> None:
    """{"candidates": []} — valid response meaning no relevant pages."""
    raw = json.dumps({"candidates": []})
    items = _parse_search_response(raw)
    assert items == []


def test_parse_invalid_json_raises() -> None:
    """Non-JSON input must raise ValueError."""
    with pytest.raises(ValueError, match="invalid JSON"):
        _parse_search_response("this is not json")


def test_parse_unextractable_dict_raises() -> None:
    """A dict with no list values and no slug key must raise ValueError."""
    raw = json.dumps({"status": "ok", "count": 3})
    with pytest.raises(ValueError, match="cannot extract candidates"):
        _parse_search_response(raw)
