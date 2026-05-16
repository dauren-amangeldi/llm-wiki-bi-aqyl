"""Unit tests for Writer Agent.

Uses vcrpy cassettes — never hits live LLM APIs. Implemented in LW-7/LW-8.
"""

import pytest


@pytest.mark.xfail(reason="Implemented in LW-7")
async def test_writer_create_page_returns_wiki_page() -> None:
    """create_page should return a WikiPage with slug, title, and content."""
    ...


@pytest.mark.xfail(reason="Implemented in LW-7")
async def test_writer_create_page_slug_is_kebab_case() -> None:
    """create_page slug must match [a-z0-9-]+ pattern."""
    ...


@pytest.mark.xfail(reason="Implemented in LW-8")
async def test_writer_update_pages_preserves_existing_content() -> None:
    """update_pages must not remove more than 40% of any existing page's content."""
    ...


@pytest.mark.xfail(reason="Implemented in LW-8")
async def test_writer_update_pages_rejects_excessive_content_drop() -> None:
    """update_pages raises ValueError when >40% content would be removed."""
    ...


@pytest.mark.xfail(reason="Implemented in LW-8")
async def test_writer_update_pages_max_five_limit() -> None:
    """update_pages raises ValueError when given more than 5 pages."""
    ...
