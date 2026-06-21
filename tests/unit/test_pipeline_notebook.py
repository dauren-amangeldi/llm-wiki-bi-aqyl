"""Tests for notebook-only ingestion (LW-N16)."""

from __future__ import annotations

from unittest.mock import MagicMock

from llm_wiki.orchestrator.pipeline import index_notebook_source


def test_index_notebook_source_uses_nb_slug_prefix() -> None:
    """Notebook sources use nb-{file_id} slug — never wiki slugs."""
    chunk_store = MagicMock()
    chunk_store.upsert_page = MagicMock()

    index_notebook_source(chunk_store, "abc-123", "Report", "Hello world content")

    chunk_store.upsert_page.assert_called_once()
    kwargs = chunk_store.upsert_page.call_args.kwargs
    assert kwargs["slug"] == "nb-abc-123"
    assert kwargs["file_id"] == "abc-123"
    assert kwargs["title"] == "Report"


def test_index_notebook_source_truncates_long_text() -> None:
    """Very long uploads are truncated before embedding."""
    chunk_store = MagicMock()
    long_text = "x" * 200_000
    index_notebook_source(chunk_store, "f1", "Big", long_text)
    content = chunk_store.upsert_page.call_args.kwargs["content"]
    assert len(content) <= 120_000
