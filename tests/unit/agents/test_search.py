"""Unit tests for Search Agent.

Uses vcrpy cassettes — never hits live LLM APIs. Implemented in LW-6.
"""

import pytest


@pytest.mark.xfail(reason="Implemented in LW-6")
async def test_search_agent_finds_relevant_pages() -> None:
    """Search Agent should return 3–10 results for a document matching wiki content."""
    ...


@pytest.mark.xfail(reason="Implemented in LW-6")
async def test_search_agent_returns_empty_for_new_topic() -> None:
    """Search Agent should return [] when no page scores >= 0.3."""
    ...


@pytest.mark.xfail(reason="Implemented in LW-6")
async def test_search_agent_empty_index() -> None:
    """Search Agent should return [] gracefully when index.md is empty."""
    ...


@pytest.mark.xfail(reason="Implemented in LW-6")
async def test_search_agent_llm_error_raises() -> None:
    """Search Agent should propagate LLM errors so the orchestrator can retry."""
    ...
