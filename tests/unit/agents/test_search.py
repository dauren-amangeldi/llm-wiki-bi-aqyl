"""Unit tests for the Search Agent v2 (LW-12).

Covers:
  - search() with all important edge cases
  - Fallback behaviour on LLM timeout
  - Hallucinated slug filtering
  - _parse_rerank_response() for all known response shapes
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_wiki.agents.search import SearchAgent, _parse_rerank_response
from llm_wiki.llm.client import LLMClient
from llm_wiki.llm.embeddings import EmbeddingStore, SearchHit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_llm(raw_response: str | object | None = None) -> LLMClient:
    """Return a mock LLMClient.

    If *raw_response* is given the mock's complete() will return it (stringified).
    """
    mock = MagicMock(spec=LLMClient)
    mock.load_prompt.return_value = "formatted prompt"
    if raw_response is not None:
        if not isinstance(raw_response, str):
            raw_response = json.dumps(raw_response)
        mock.complete = AsyncMock(return_value=(raw_response, MagicMock()))
    return mock  # type: ignore[return-value]


def _mock_store(
    count: int = 5,
    query_results: list[SearchHit] | None = None,
) -> EmbeddingStore:
    """Return a mock EmbeddingStore."""
    mock = MagicMock(spec=EmbeddingStore)
    mock.count.return_value = count
    if query_results is not None:
        mock.query.return_value = query_results
    else:
        mock.query.return_value = [
            SearchHit(slug=f"page-{i}", title=f"Page {i}", section="General", similarity=0.9 - i * 0.05)
            for i in range(min(count, 5))
        ]
    return mock  # type: ignore[return-value]


def _hit(slug: str, score: float = 0.9, section: str = "General") -> SearchHit:
    return SearchHit(slug=slug, title=slug.replace("-", " ").title(), section=section, similarity=score)


def _rerank_resp(*slugs_scores: tuple[str, float]) -> str:
    """Build a canonical rerank JSON response."""
    hits = [{"slug": s, "rerank_score": r, "reason": f"reason for {s}"} for s, r in slugs_scores]
    return json.dumps({"hits": hits})


# ===========================================================================
# search() edge cases
# ===========================================================================


async def test_empty_wiki_returns_empty_no_llm_call() -> None:
    """Empty EmbeddingStore → [] without calling the LLM."""
    llm = _mock_llm()
    store = _mock_store(count=0)
    agent = SearchAgent(llm, store)

    results = await agent.search("some text about AI", file_id="f1")

    assert results == []
    llm.complete.assert_not_called()  # type: ignore[attr-defined]


async def test_all_below_threshold_returns_empty_no_llm_call() -> None:
    """All candidates below similarity threshold → [] without LLM call."""
    llm = _mock_llm()
    # All similarities below default threshold of 0.3
    low_hits = [_hit(f"page-{i}", score=0.1) for i in range(3)]
    store = _mock_store(count=3, query_results=low_hits)

    agent = SearchAgent(llm, store)

    with MagicMock() as settings_mock:
        import llm_wiki.agents.search as search_module
        original_settings = search_module.settings
        search_module.settings = MagicMock(
            search_top_k=20,
            search_similarity_threshold=0.3,
            search_final_k_max=10,
            search_summary_max_chars=8000,
            wiki_language="en",
        )
        results = await agent.search("unrelated content", file_id="f2")
        search_module.settings = original_settings

    assert results == []
    llm.complete.assert_not_called()  # type: ignore[attr-defined]


async def test_happy_path_returns_sorted_hits() -> None:
    """20 candidates from embedding → LLM returns 5 reranked hits in order."""
    candidates = [_hit(f"page-{i}", score=0.8 - i * 0.01) for i in range(20)]
    llm = _mock_llm(_rerank_resp(
        ("page-3", 0.95),
        ("page-0", 0.90),
        ("page-7", 0.85),
        ("page-1", 0.80),
        ("page-5", 0.75),
    ))
    store = _mock_store(count=20, query_results=candidates)

    import llm_wiki.agents.search as search_module
    original_settings = search_module.settings
    search_module.settings = MagicMock(
        search_top_k=20,
        search_similarity_threshold=0.3,
        search_final_k_max=10,
        search_summary_max_chars=8000,
        wiki_language="en",
    )
    try:
        agent = SearchAgent(llm, store)
        results = await agent.search("file content", file_id="f3")
    finally:
        search_module.settings = original_settings

    assert len(results) == 5
    scores = [r.rerank_score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0].slug == "page-3"


async def test_llm_returns_more_than_max_trimmed() -> None:
    """LLM returns more hits than SEARCH_FINAL_K_MAX → silently trimmed."""
    candidates = [_hit(f"page-{i}", score=0.9) for i in range(15)]
    hits_from_llm = [(f"page-{i}", 0.9 - i * 0.01) for i in range(15)]
    llm = _mock_llm(_rerank_resp(*hits_from_llm))
    store = _mock_store(count=15, query_results=candidates)

    import llm_wiki.agents.search as search_module
    original = search_module.settings
    search_module.settings = MagicMock(
        search_top_k=20,
        search_similarity_threshold=0.3,
        search_final_k_max=5,  # max=5
        search_summary_max_chars=8000,
        wiki_language="en",
    )
    try:
        agent = SearchAgent(llm, store)
        results = await agent.search("text", file_id="f4")
    finally:
        search_module.settings = original

    assert len(results) <= 5


async def test_llm_returns_zero_hits_valid() -> None:
    """LLM can legitimately return empty hits — not an error."""
    candidates = [_hit("page-0", score=0.9)]
    llm = _mock_llm(json.dumps({"hits": []}))
    store = _mock_store(count=1, query_results=candidates)

    import llm_wiki.agents.search as search_module
    original = search_module.settings
    search_module.settings = MagicMock(
        search_top_k=20,
        search_similarity_threshold=0.3,
        search_final_k_max=10,
        search_summary_max_chars=8000,
        wiki_language="en",
    )
    try:
        agent = SearchAgent(llm, store)
        results = await agent.search("unrelated text", file_id="f5")
    finally:
        search_module.settings = original

    assert results == []


async def test_llm_returns_unknown_slug_filtered() -> None:
    """Hallucinated slugs from LLM that weren't in candidates are silently dropped."""
    candidates = [_hit("real-page", score=0.9)]
    llm = _mock_llm(json.dumps({"hits": [
        {"slug": "hallucinated-slug", "rerank_score": 0.99, "reason": "invented"},
        {"slug": "real-page", "rerank_score": 0.85, "reason": "valid"},
    ]}))
    store = _mock_store(count=1, query_results=candidates)

    import llm_wiki.agents.search as search_module
    original = search_module.settings
    search_module.settings = MagicMock(
        search_top_k=20,
        search_similarity_threshold=0.3,
        search_final_k_max=10,
        search_summary_max_chars=8000,
        wiki_language="en",
    )
    try:
        agent = SearchAgent(llm, store)
        results = await agent.search("text", file_id="f6")
    finally:
        search_module.settings = original

    slugs = [r.slug for r in results]
    assert "hallucinated-slug" not in slugs
    assert "real-page" in slugs


async def test_llm_timeout_fallback_returns_top_5_embedding() -> None:
    """LLM raises TimeoutError → fallback returns top-5 by similarity, rerank_score=None."""
    candidates = [_hit(f"page-{i}", score=0.9 - i * 0.05) for i in range(10)]
    llm = _mock_llm()
    llm.complete = AsyncMock(side_effect=TimeoutError("LLM timeout"))
    store = _mock_store(count=10, query_results=candidates)

    import llm_wiki.agents.search as search_module
    original = search_module.settings
    search_module.settings = MagicMock(
        search_top_k=20,
        search_similarity_threshold=0.0,  # all above threshold
        search_final_k_max=10,
        search_summary_max_chars=8000,
        wiki_language="en",
    )
    try:
        agent = SearchAgent(llm, store)
        results = await agent.search("text", file_id="f7")
    finally:
        search_module.settings = original

    assert len(results) <= 5
    for r in results:
        assert r.rerank_score is None


async def test_run_shim_delegates_to_search() -> None:
    """SearchAgent.run() produces the same result as search() (shim compatibility)."""
    candidates = [_hit("page-0", score=0.9)]
    llm = _mock_llm(_rerank_resp(("page-0", 0.9)))
    store = _mock_store(count=1, query_results=candidates)

    import llm_wiki.agents.search as search_module
    original = search_module.settings
    search_module.settings = MagicMock(
        search_top_k=20,
        search_similarity_threshold=0.3,
        search_final_k_max=10,
        search_summary_max_chars=8000,
        wiki_language="en",
    )
    try:
        agent = SearchAgent(llm, store)
        via_run = await agent.run("text", ["old heading list ignored"], file_id="f8")
        via_search = await agent.search("text", file_id="f8")
    finally:
        search_module.settings = original

    assert [r.slug for r in via_run] == [r.slug for r in via_search]


# ===========================================================================
# _parse_rerank_response — unit tests for all accepted shapes
# ===========================================================================


def test_parse_hits_wrapper() -> None:
    """{"hits": [...]} — canonical format."""
    raw = json.dumps({"hits": [{"slug": "x", "rerank_score": 0.8, "reason": "ok"}]})
    items = _parse_rerank_response(raw)
    assert items[0]["slug"] == "x"


def test_parse_candidates_alias() -> None:
    """{"candidates": [...]} — common alias."""
    raw = json.dumps({"candidates": [{"slug": "y", "rerank_score": 0.7}]})
    items = _parse_rerank_response(raw)
    assert items[0]["slug"] == "y"


def test_parse_results_alias() -> None:
    """{"results": [...]} — second common alias."""
    raw = json.dumps({"results": [{"slug": "z", "rerank_score": 0.6}]})
    items = _parse_rerank_response(raw)
    assert items[0]["slug"] == "z"


def test_parse_bare_array() -> None:
    """[...] — bare array accepted as a fallback response shape."""
    raw = json.dumps([{"slug": "a", "rerank_score": 0.9}])
    items = _parse_rerank_response(raw)
    assert items[0]["slug"] == "a"


def test_parse_markdown_fenced_json() -> None:
    """```json\\n{...}\\n``` — markdown fence stripped."""
    payload = {"hits": [{"slug": "fenced", "rerank_score": 0.8}]}
    raw = "```json\n" + json.dumps(payload) + "\n```"
    items = _parse_rerank_response(raw)
    assert items[0]["slug"] == "fenced"


def test_parse_single_flat_object() -> None:
    """{"slug": ..., ...} — flat dict wraps itself as a single-item list."""
    raw = json.dumps({"slug": "solo", "rerank_score": 0.7, "reason": "only one"})
    items = _parse_rerank_response(raw)
    assert items[0]["slug"] == "solo"


def test_parse_empty_hits_list() -> None:
    """{"hits": []} — valid empty response."""
    items = _parse_rerank_response(json.dumps({"hits": []}))
    assert items == []


def test_parse_invalid_json_raises() -> None:
    """Non-JSON string must raise ValueError."""
    with pytest.raises(ValueError, match="invalid JSON"):
        _parse_rerank_response("not json at all")


def test_parse_unextractable_dict_raises() -> None:
    """Dict without list values and no slug key raises ValueError."""
    with pytest.raises(ValueError):
        _parse_rerank_response(json.dumps({"status": "ok", "count": 3}))
