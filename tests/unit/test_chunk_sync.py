"""Unit tests for storage/chunk_sync.py (LW-20.1)."""

from __future__ import annotations

from unittest.mock import MagicMock

from llm_wiki.storage.chunk_sync import sync_chunks_for_page


def test_sync_noop_when_store_is_none() -> None:
    """sync_chunks_for_page is a complete no-op when chunk_store=None."""
    # We pass a sentinel that would raise if any attribute is accessed
    sentinel = MagicMock(spec=[])  # no attributes allowed
    sync_chunks_for_page(chunk_store=None, slug="s", title="T", content="c")
    sentinel.upsert_page.assert_not_called()


def test_sync_calls_upsert_page() -> None:
    """sync_chunks_for_page delegates to chunk_store.upsert_page."""
    store = MagicMock()
    sync_chunks_for_page(
        chunk_store=store,
        slug="my-page",
        title="My Page",
        content="## Section\n\nContent.",
        file_id="fid-001",
    )
    store.upsert_page.assert_called_once_with(
        slug="my-page",
        title="My Page",
        content="## Section\n\nContent.",
        file_id="fid-001",
    )


def test_sync_swallows_exception() -> None:
    """sync_chunks_for_page does NOT propagate exceptions from the store."""
    store = MagicMock()
    store.upsert_page.side_effect = RuntimeError("Chroma is down")
    # Must not raise
    sync_chunks_for_page(chunk_store=store, slug="s", title="T", content="c")
