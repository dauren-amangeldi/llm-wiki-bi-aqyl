"""Unit tests for the Twins case-context loader."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from llm_wiki.agents.twins import load_case_context
from llm_wiki.storage.metadata import FileRecord


def _file_record(created_pages: list[str], updated_pages: list[str] | None = None) -> FileRecord:
    fr = MagicMock(spec=FileRecord)
    fr.created_pages = created_pages
    fr.updated_pages = updated_pages or []
    return fr


def test_load_case_context_dedupes_slugs_across_documents() -> None:
    docs = [
        _file_record(created_pages=["page-a"]),
        _file_record(created_pages=["page-a"], updated_pages=["page-b"]),
    ]
    with patch("llm_wiki.storage.object_store.get_object_store") as mock_store_factory:
        mock_store = MagicMock()
        mock_store.get_text.side_effect = lambda key: f"body for {key}"
        mock_store_factory.return_value = mock_store

        context = load_case_context(docs)

    assert context.count("page-a") >= 1
    assert "page-b" in context
    assert mock_store.get_text.call_count == 2  # page-a loaded once, not twice


def test_load_case_context_skips_empty_bodies() -> None:
    docs = [_file_record(created_pages=["empty-page"])]
    with patch("llm_wiki.storage.object_store.get_object_store") as mock_store_factory:
        mock_store = MagicMock()
        mock_store.get_text.return_value = ""
        mock_store_factory.return_value = mock_store

        context = load_case_context(docs)

    assert context == ""


def test_load_case_context_returns_empty_string_for_no_documents() -> None:
    assert load_case_context([]) == ""
