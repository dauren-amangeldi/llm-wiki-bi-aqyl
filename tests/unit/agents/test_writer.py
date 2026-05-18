"""Unit tests for the Writer Agent (LW-7 / LW-8)."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_wiki.agents.writer import WikiPage, WriterAgent
from llm_wiki.llm.client import LLMClient


def _mock_llm_for_create(slug: str, title: str, content: str) -> LLMClient:
    """Return a mock LLM whose complete() yields a create-page JSON response."""
    mock = MagicMock(spec=LLMClient)
    mock.load_prompt.return_value = "prompt"
    mock.complete = AsyncMock(
        return_value=(
            json.dumps({"slug": slug, "title": title, "content": content}),
            MagicMock(),
        )
    )
    return mock  # type: ignore[return-value]


def _mock_llm_for_update(title: str, content: str) -> LLMClient:
    """Return a mock LLM whose complete() yields an update-page JSON response."""
    mock = MagicMock(spec=LLMClient)
    mock.load_prompt.return_value = "prompt"
    mock.complete = AsyncMock(
        return_value=(
            json.dumps({"title": title, "content": content}),
            MagicMock(),
        )
    )
    return mock  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# LW-7 — create_page
# ---------------------------------------------------------------------------


async def test_writer_create_page_returns_wiki_page() -> None:
    """create_page returns a WikiPage with correct slug, title, and content."""
    llm = _mock_llm_for_create("transformer-architecture", "Transformer Architecture", "# Transformers\n\nContent.")
    agent = WriterAgent(llm)

    page = await agent.create_page("raw text about transformers", file_id="f1")

    assert isinstance(page, WikiPage)
    assert page.slug == "transformer-architecture"
    assert page.title == "Transformer Architecture"
    assert page.content.strip() != ""
    assert "f1" in page.source_files


async def test_writer_create_page_slug_is_kebab_case() -> None:
    """create_page slug must match [a-z0-9][a-z0-9-]* pattern."""
    llm = _mock_llm_for_create("my-new-page", "My New Page", "Some content here.")
    agent = WriterAgent(llm)

    page = await agent.create_page("raw", file_id="f2")

    import re
    assert re.fullmatch(r"[a-z0-9][a-z0-9-]*[a-z0-9]", page.slug)


async def test_writer_create_page_derives_slug_from_title() -> None:
    """create_page derives slug from title when LLM returns an invalid slug."""
    llm = _mock_llm_for_create("INVALID SLUG!!!", "Transformer Architecture", "Content here.")
    agent = WriterAgent(llm)

    page = await agent.create_page("raw", file_id="f3")

    # Should have derived "transformer-architecture" from the title
    assert page.slug == "transformer-architecture"


async def test_writer_create_page_empty_content_raises() -> None:
    """create_page raises ValueError when LLM returns empty content."""
    mock = MagicMock(spec=LLMClient)
    mock.load_prompt.return_value = "prompt"
    mock.complete = AsyncMock(
        return_value=(
            json.dumps({"slug": "valid-slug", "title": "Title", "content": "   "}),
            MagicMock(),
        )
    )
    agent = WriterAgent(mock)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="empty content"):
        await agent.create_page("raw", file_id="f4")


async def test_writer_create_page_invalid_json_raises() -> None:
    """create_page raises ValueError on non-JSON LLM response."""
    mock = MagicMock(spec=LLMClient)
    mock.load_prompt.return_value = "prompt"
    mock.complete = AsyncMock(return_value=("not json", MagicMock()))
    agent = WriterAgent(mock)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="invalid JSON"):
        await agent.create_page("raw", file_id="f5")


# ---------------------------------------------------------------------------
# LW-8 — update_pages
# ---------------------------------------------------------------------------


async def test_writer_update_pages_preserves_existing_content() -> None:
    """update_pages does not reject updates that keep ≥60% of original content."""
    original = "A" * 1000
    new = "A" * 700  # 30% drop — within the 40% limit
    llm = _mock_llm_for_update("Updated Title", new)
    agent = WriterAgent(llm)

    pages = [WikiPage(slug="my-page", title="My Page", content=original)]
    updated = await agent.update_pages("raw content", pages, file_id="f6")

    assert len(updated) == 1
    assert updated[0].slug == "my-page"
    assert updated[0].content == new


async def test_writer_update_pages_rejects_excessive_content_drop() -> None:
    """update_pages raises ValueError when >40% of content would be removed."""
    original = "A" * 1000
    new = "A" * 500  # 50% drop — exceeds the 40% limit
    llm = _mock_llm_for_update("Title", new)
    agent = WriterAgent(llm)

    pages = [WikiPage(slug="my-page", title="My Page", content=original)]
    with pytest.raises(ValueError, match="would remove"):
        await agent.update_pages("raw", pages, file_id="f7")


async def test_writer_update_pages_max_five_limit() -> None:
    """update_pages raises ValueError when given more than 5 pages."""
    llm = MagicMock(spec=LLMClient)
    agent = WriterAgent(llm)

    pages = [
        WikiPage(slug=f"page-{i}", title=f"Page {i}", content="content")
        for i in range(6)
    ]
    with pytest.raises(ValueError, match="at most 5 pages"):
        await agent.update_pages("raw", pages, file_id="f8")


async def test_writer_update_pages_merges_source_files() -> None:
    """update_pages appends the new file_id to source_files without duplicates."""
    original = "x" * 500
    new = "x" * 450
    llm = _mock_llm_for_update("Title", new)
    agent = WriterAgent(llm)

    pages = [WikiPage(slug="p", title="P", content=original, source_files=["old-file"])]
    updated = await agent.update_pages("raw", pages, file_id="new-file")

    assert "old-file" in updated[0].source_files
    assert "new-file" in updated[0].source_files


async def test_writer_check_content_drop_false_for_empty_original() -> None:
    """_check_content_drop returns False when original content is empty."""
    agent = WriterAgent(MagicMock(spec=LLMClient))
    # Empty original — no drop is possible
    assert agent._check_content_drop("", "anything") is False  # noqa: SLF001
