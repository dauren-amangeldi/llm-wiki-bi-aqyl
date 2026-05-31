"""Unit tests for EmbeddingStore (LW-11).

All tests use an in-memory ChromaDB client (EphemeralClient) and a mocked
LLMClient so no real API calls are made.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import chromadb
import pytest

from llm_wiki.llm.embeddings import (
    EmbeddingError,
    EmbeddingModelMismatchError,
    EmbeddingStore,
    HeadingInfo,
    SearchHit,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIM = 8  # small dimension for fast tests


def _fake_embed(texts: list[str], **_kwargs: Any) -> list[list[float]]:
    """Return deterministic unit-length embeddings based on text hash."""
    import hashlib

    results = []
    for t in texts:
        seed = int(hashlib.md5(t.encode()).hexdigest(), 16) % (10**6)
        vec = [float((seed >> i) & 1) for i in range(_DIM)]
        # Normalise to unit length (avoid zero vector)
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        results.append([v / norm for v in vec])
    return results


def _mock_llm() -> MagicMock:
    mock = MagicMock()
    mock.embed.side_effect = _fake_embed
    return mock


def _store(mock_llm: MagicMock | None = None, dim: int = _DIM) -> EmbeddingStore:
    """Return an EmbeddingStore backed by an in-memory ChromaDB client."""
    with patch("llm_wiki.llm.embeddings.EmbeddingStore.__init__.__wrapped__", None):
        pass

    with (
        patch("llm_wiki.config.settings") as mock_settings,
    ):
        mock_settings.embedding_model = "text-embedding-3-small"
        mock_settings.embedding_dimensions = dim
        mock_settings.embedding_batch_size = 50

        return EmbeddingStore(
            chroma_path=Path("/tmp/unused"),
            llm_client=mock_llm or _mock_llm(),
            chroma_client=chromadb.EphemeralClient(),
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_upsert_single_then_query() -> None:
    """upsert_heading followed by query(same_title) returns score close to 1."""
    llm = _mock_llm()
    store = _store(llm)

    store.upsert_heading(slug="transformers", title="Transformers", section="ML", level=2)

    hits = store.query("Transformers", top_k=5)

    assert len(hits) == 1
    assert hits[0].slug == "transformers"
    assert hits[0].title == "Transformers"
    assert hits[0].similarity > 0.95  # same text → near-identical embedding


def test_upsert_then_delete_not_returned() -> None:
    """After delete(), query no longer returns the removed heading."""
    store = _store()

    store.upsert_heading(slug="bert", title="BERT", section="NLP", level=2)
    assert store.count() == 1

    store.delete("bert")
    assert store.count() == 0

    hits = store.query("BERT", top_k=5)
    slugs = [h.slug for h in hits]
    assert "bert" not in slugs


def test_batch_upsert_count_correct() -> None:
    """upsert_many inserts the expected number of headings."""
    store = _store()

    headings = [
        HeadingInfo(slug=f"page-{i}", title=f"Page {i}", section="General", level=2)
        for i in range(20)
    ]
    store.upsert_many(headings)

    assert store.count() == 20


def test_query_top_k_respected() -> None:
    """query() returns at most top_k results."""
    store = _store()

    for i in range(10):
        store.upsert_heading(
            slug=f"p{i}", title=f"Topic {i}", section="General", level=2
        )

    hits = store.query("Topic", top_k=3)
    assert len(hits) <= 3


def test_idempotent_upsert_no_duplicate() -> None:
    """Calling upsert_heading twice with the same slug does not create a duplicate."""
    store = _store()

    store.upsert_heading(slug="ml", title="Machine Learning", section="AI", level=2)
    store.upsert_heading(slug="ml", title="Machine Learning v2", section="AI", level=2)

    assert store.count() == 1
    # The latest title should win
    hits = store.query("Machine Learning v2", top_k=1)
    assert hits[0].slug == "ml"


def test_query_empty_collection_returns_empty() -> None:
    """query() on an empty collection returns []."""
    store = _store()
    assert store.query("anything") == []


def test_update_metadata_changes_section() -> None:
    """update_metadata changes the section without re-embedding."""
    llm = _mock_llm()
    store = _store(llm)

    store.upsert_heading(slug="rl", title="Reinforcement Learning", section="Old", level=2)
    embed_call_count_before = llm.embed.call_count

    store.update_metadata(slug="rl", section="New Section")

    # No new embed call should have been made
    assert llm.embed.call_count == embed_call_count_before


def test_usage_logged_after_embed(tmp_path: Path) -> None:
    """embed() writes a record to usage.log with the correct model."""
    from llm_wiki.llm.client import LLMClient

    usage_log = tmp_path / "usage.log"

    # Patch settings so LLMClient writes to tmp usage.log
    with (
        patch("llm_wiki.llm.client.settings") as cfg,
        patch("llm_wiki.config.settings") as cfg2,
    ):
        cfg.openai_api_key = "test-key"
        cfg.embedding_model = "text-embedding-3-small"
        cfg.embedding_batch_size = 50
        cfg.embedding_dimensions = 1536
        cfg.usage_log_path = usage_log
        cfg.llm_provider = "openai"
        cfg.openai_model = "gpt-5.4-mini"
        cfg.price_table = {
            "text-embedding-3-small": {"input": 0.02, "output": 0.0},
        }
        cfg2.openai_api_key = "test-key"
        cfg2.embedding_model = "text-embedding-3-small"
        cfg2.embedding_dimensions = 1536
        cfg2.embedding_batch_size = 50
        cfg2.price_table = {
            "text-embedding-3-small": {"input": 0.02, "output": 0.0},
        }

        fake_response = MagicMock()
        fake_response.data = [MagicMock(embedding=[0.1] * 1536)]
        fake_response.usage.total_tokens = 5

        client = LLMClient.__new__(LLMClient)
        client._provider = "openai"
        client._usage_log_path = usage_log
        client._model = "gpt-5.4-mini"
        client._non_retryable = (Exception,)  # won't match real errors

        import openai

        sync_oa = MagicMock(spec=openai.OpenAI)
        sync_oa.embeddings.create.return_value = fake_response

        with patch("llm_wiki.llm.client.openai.OpenAI", return_value=sync_oa):
            result = client.embed(["hello world"])

    assert result == [[0.1] * 1536]
    assert usage_log.exists()
    lines = [json.loads(ln) for ln in usage_log.read_text().splitlines() if ln]
    assert lines[0]["agent_type"] == "embed"
    assert lines[0]["model"] == "text-embedding-3-small"


def test_chroma_failure_does_not_break_index(tmp_path: Path) -> None:
    """If EmbeddingStore.upsert_heading raises, IndexStorage.add_page still writes."""
    from llm_wiki.storage.index import IndexStorage

    index_path = tmp_path / "index.md"
    index_path.write_text("# Wiki Index\n\n## General\n\n")

    broken_store = MagicMock()
    broken_store.upsert_heading.side_effect = RuntimeError("Chroma is down")

    storage = IndexStorage(index_path=index_path, embedding_store=broken_store)
    # Must NOT raise even though embedding store throws
    storage.add_page("test-page", "General")

    assert "[[test-page]]" in index_path.read_text()


def test_model_mismatch_raises_on_init() -> None:
    """EmbeddingStore raises EmbeddingModelMismatchError if stored model differs."""
    from unittest.mock import patch

    ephemeral = chromadb.EphemeralClient()

    # First creation stores "old-model"
    with patch("llm_wiki.config.settings") as cfg:
        cfg.embedding_model = "old-model"
        cfg.embedding_dimensions = 8
        cfg.embedding_batch_size = 50
        EmbeddingStore(
            chroma_path=Path("/tmp/unused"),
            llm_client=_mock_llm(),
            chroma_client=ephemeral,
        )

    # Second creation with different model should raise
    with (
        patch("llm_wiki.config.settings") as cfg,
        pytest.raises(EmbeddingModelMismatchError),
    ):
        cfg.embedding_model = "new-model"
        cfg.embedding_dimensions = 8
        cfg.embedding_batch_size = 50
        EmbeddingStore(
            chroma_path=Path("/tmp/unused"),
            llm_client=_mock_llm(),
            chroma_client=ephemeral,
        )


def test_clear_resets_collection() -> None:
    """clear() removes all entries and allows fresh insertion."""
    store = _store()

    for i in range(5):
        store.upsert_heading(slug=f"s{i}", title=f"T{i}", section="X", level=2)
    assert store.count() == 5

    store.clear()
    assert store.count() == 0

    store.upsert_heading(slug="new", title="New Page", section="X", level=2)
    assert store.count() == 1


def test_search_hit_is_frozen() -> None:
    """SearchHit is a frozen dataclass — immutable after construction."""
    hit = SearchHit(slug="x", title="X", section="General", similarity=0.9)
    with pytest.raises(Exception):
        hit.slug = "y"  # type: ignore[misc]
