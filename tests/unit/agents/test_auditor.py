"""Unit tests for agents/auditor.py (LW-15).

These tests mock the LLMClient so they run fully offline — no API key or
network access required.

NOTE: vcrpy cassettes for real-API tests live in
``tests/fixtures/vcr_cassettes/auditor_*``.  The tests in this file focus
on the parsing, chunking, and filtering logic using controlled mock responses.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_wiki.agents.auditor import AuditorAgent
from llm_wiki.quality.models import Issue, IssueKind, IssueSection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm(response: str | Exception = "[]") -> MagicMock:
    """Return a mock LLMClient whose ``complete`` returns *response*."""
    mock = MagicMock()
    if isinstance(response, Exception):
        mock.complete = AsyncMock(side_effect=response)
    else:
        mock.complete = AsyncMock(return_value=response)
    return mock


def _pages(n: int) -> list[tuple[str, str]]:
    """Generate *n* dummy wiki pages."""
    return [(f"page-{i:03d}", f"# Page {i}\nContent for page {i}.") for i in range(n)]


# ---------------------------------------------------------------------------
# Basic response parsing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_wiki_returns_empty() -> None:
    agent = AuditorAgent(_make_llm())
    issues = await agent.run(wiki_pages=[], mode="sync")
    assert issues == []


@pytest.mark.asyncio
async def test_clean_wiki_no_issues() -> None:
    agent = AuditorAgent(_make_llm("[]"))
    pages = _pages(3)
    issues = await agent.run(wiki_pages=pages, mode="sync")
    assert issues == []


@pytest.mark.asyncio
async def test_contradiction_detected() -> None:
    response_json = json.dumps([{
        "kind": "contradiction",
        "page_slug": "page-000",
        "description": "Contradicts page-001 on the version number.",
        "related_slugs": ["page-001"],
    }])
    pages = _pages(2)
    agent = AuditorAgent(_make_llm(response_json))
    issues = await agent.run(wiki_pages=pages, mode="sync")
    assert len(issues) == 1
    assert issues[0].kind == IssueKind.CONTRADICTION
    assert issues[0].page_slug == "page-000"
    assert "page-001" in issues[0].related_slugs


@pytest.mark.asyncio
async def test_duplicate_detected() -> None:
    response_json = json.dumps([{
        "kind": "duplicate",
        "page_slug": "page-000",
        "description": "Nearly identical to page-001.",
        "related_slugs": ["page-001"],
    }])
    pages = _pages(2)
    agent = AuditorAgent(_make_llm(response_json))
    issues = await agent.run(wiki_pages=pages, mode="sync")
    assert issues[0].kind == IssueKind.DUPLICATE


@pytest.mark.asyncio
async def test_suspected_stale_detected() -> None:
    response_json = json.dumps([{
        "kind": "suspected_stale",
        "page_slug": "page-000",
        "description": "Describes v1 as current; v5 likely released since.",
        "related_slugs": [],
    }])
    pages = _pages(2)
    agent = AuditorAgent(_make_llm(response_json))
    issues = await agent.run(wiki_pages=pages, mode="sync")
    assert issues[0].kind == IssueKind.SUSPECTED_STALE


# ---------------------------------------------------------------------------
# Kind filtering
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_issue_kind_filtering_removes_invalid_kinds() -> None:
    """LLM accidentally returned dead_link (not an auditor kind) — filter out."""
    response_json = json.dumps([
        {
            "kind": "dead_link",
            "page_slug": "page-000",
            "description": "Should be filtered.",
            "related_slugs": [],
        },
        {
            "kind": "contradiction",
            "page_slug": "page-000",
            "description": "Valid auditor issue.",
            "related_slugs": [],
        },
    ])
    pages = _pages(3)
    agent = AuditorAgent(_make_llm(response_json))
    issues = await agent.run(wiki_pages=pages, mode="sync")
    assert len(issues) == 1
    assert issues[0].kind == IssueKind.CONTRADICTION


@pytest.mark.asyncio
async def test_hallucinated_slug_filtered() -> None:
    """LLM returned a slug that was not in the input — must be filtered."""
    response_json = json.dumps([{
        "kind": "contradiction",
        "page_slug": "invented-slug-xyz",
        "description": "Hallucinated slug.",
        "related_slugs": [],
    }])
    pages = _pages(2)
    agent = AuditorAgent(_make_llm(response_json))
    issues = await agent.run(wiki_pages=pages, mode="sync")
    assert issues == []


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_batching_respected() -> None:
    """110 pages at BATCH_SIZE=50 → exactly 3 LLM calls."""
    llm = _make_llm("[]")
    agent = AuditorAgent(llm)
    agent.BATCH_SIZE = 50
    await agent.run(wiki_pages=_pages(110), mode="sync")
    assert llm.complete.call_count == 3


@pytest.mark.asyncio
async def test_batching_single_page() -> None:
    """1 page → exactly 1 LLM call."""
    llm = _make_llm("[]")
    agent = AuditorAgent(llm)
    await agent.run(wiki_pages=_pages(1), mode="sync")
    assert llm.complete.call_count == 1


@pytest.mark.asyncio
async def test_batching_exact_multiple() -> None:
    """100 pages at BATCH_SIZE=50 → exactly 2 LLM calls."""
    llm = _make_llm("[]")
    agent = AuditorAgent(llm)
    agent.BATCH_SIZE = 50
    await agent.run(wiki_pages=_pages(100), mode="sync")
    assert llm.complete.call_count == 2


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_deduplication_across_batches() -> None:
    """Same (kind, page_slug, description) from two batches → deduplicated."""
    same_response = json.dumps([{
        "kind": "suspected_stale",
        "page_slug": "page-000",
        "description": "Possibly outdated.",
        "related_slugs": [],
    }])
    llm = _make_llm(same_response)
    agent = AuditorAgent(llm)
    agent.BATCH_SIZE = 1  # force 2 batches for 2 pages
    pages = [("page-000", "content 0"), ("page-000", "content 1")]
    # page-000 appears in both batches → only one unique issue
    issues = await agent.run(wiki_pages=pages, mode="sync")
    assert len(issues) == 1


# ---------------------------------------------------------------------------
# Error resilience
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_error_returns_empty_for_chunk() -> None:
    """If the LLM call fails, that chunk returns [] and the run continues."""
    llm = _make_llm(RuntimeError("API down"))
    agent = AuditorAgent(llm)
    issues = await agent.run(wiki_pages=_pages(3), mode="sync")
    assert issues == []


@pytest.mark.asyncio
async def test_json_parse_error_returns_empty() -> None:
    """Malformed JSON from LLM → empty list, no exception."""
    llm = _make_llm("this is not json at all")
    agent = AuditorAgent(llm)
    issues = await agent.run(wiki_pages=_pages(2), mode="sync")
    assert issues == []


# ---------------------------------------------------------------------------
# JSON wrapper shapes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("wrapper", [
    # Wrapped in {"issues": [...]}
    lambda raw: json.dumps({"issues": raw}),
    # Wrapped in ```json ... ```
    lambda raw: f"```json\n{json.dumps(raw)}\n```",
    # Raw array
    lambda raw: json.dumps(raw),
])
@pytest.mark.asyncio
async def test_various_response_shapes(wrapper: Any) -> None:
    raw = [{
        "kind": "duplicate",
        "page_slug": "page-000",
        "description": "Duplicate of page-001.",
        "related_slugs": ["page-001"],
    }]
    response_text = wrapper(raw)
    pages = _pages(2)
    agent = AuditorAgent(_make_llm(response_text))
    issues = await agent.run(wiki_pages=pages, mode="sync")
    assert len(issues) == 1
    assert issues[0].kind == IssueKind.DUPLICATE


# ---------------------------------------------------------------------------
# Section
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_issues_have_llm_flagged_section() -> None:
    response_json = json.dumps([{
        "kind": "contradiction",
        "page_slug": "page-000",
        "description": "Contradicts something.",
    }])
    pages = _pages(2)
    agent = AuditorAgent(_make_llm(response_json))
    issues = await agent.run(wiki_pages=pages, mode="sync")
    assert all(i.section == IssueSection.LLM_FLAGGED for i in issues)
