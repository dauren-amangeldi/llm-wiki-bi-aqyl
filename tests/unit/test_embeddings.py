"""Unit tests for EmbeddingStore (pgvector-backed).

Tests use the ``vector_engine`` fixture (a sync engine to the test Postgres
with the pgvector extension) and a mocked LLMClient so no real API calls are
made. Vectors are full-dimension (config's embedding_dimensions) because the
``vector()`` column dimension is fixed at import time.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llm_wiki.config import settings
from llm_wiki.llm.embeddings import (
    EmbeddingModelMismatchError,
    EmbeddingStore,
    HeadingInfo,
    SearchHit,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIM = settings.embedding_dimensions  # must match the vector() column dimension


def _fake_embed(texts: list[str], **_kwargs: Any) -> list[list[float]]:
    """Return deterministic unit-length embeddings derived from an MD5 digest.

    Identical text → identical vector (so a query for the same text scores ~1.0);
    different text → different vector.
    """
    results = []
    for t in texts:
        digest = hashlib.md5(t.encode()).digest()  # 16 stable bytes
        vec = [float(digest[i % len(digest)]) for i in range(_DIM)]
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        results.append([v / norm for v in vec])
    return results


def _mock_llm() -> MagicMock:
    mock = MagicMock()
    mock.embed.side_effect = _fake_embed
    return mock


def _store(vector_engine: Any, mock_llm: MagicMock | None = None) -> EmbeddingStore:
    """Return an EmbeddingStore bound to the test Postgres engine."""
    return EmbeddingStore(llm_client=mock_llm or _mock_llm(), engine=vector_engine)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_upsert_single_then_query(vector_engine: Any) -> None:
    """upsert_heading followed by query(same_title) returns similarity close to 1."""
    llm = _mock_llm()
    store = _store(vector_engine, llm)

    store.upsert_heading(slug="transformers", title="Transformers", section="ML", level=2)

    hits = store.query("Transformers", top_k=5)

    assert len(hits) == 1
    assert hits[0].slug == "transformers"
    assert hits[0].title == "Transformers"
    assert hits[0].similarity > 0.95  # same text → near-identical embedding


def test_upsert_then_delete_not_returned(vector_engine: Any) -> None:
    """After delete(), query no longer returns the removed heading."""
    store = _store(vector_engine)

    store.upsert_heading(slug="bert", title="BERT", section="NLP", level=2)
    assert store.count() == 1

    store.delete("bert")
    assert store.count() == 0

    hits = store.query("BERT", top_k=5)
    slugs = [h.slug for h in hits]
    assert "bert" not in slugs


def test_batch_upsert_count_correct(vector_engine: Any) -> None:
    """upsert_many inserts the expected number of headings."""
    store = _store(vector_engine)

    headings = [
        HeadingInfo(slug=f"page-{i}", title=f"Page {i}", section="General", level=2)
        for i in range(20)
    ]
    store.upsert_many(headings)

    assert store.count() == 20


def test_query_top_k_respected(vector_engine: Any) -> None:
    """query() returns at most top_k results."""
    store = _store(vector_engine)

    for i in range(10):
        store.upsert_heading(slug=f"p{i}", title=f"Topic {i}", section="General", level=2)

    hits = store.query("Topic", top_k=3)
    assert len(hits) <= 3


def test_idempotent_upsert_no_duplicate(vector_engine: Any) -> None:
    """Calling upsert_heading twice with the same slug does not create a duplicate."""
    store = _store(vector_engine)

    store.upsert_heading(slug="ml", title="Machine Learning", section="AI", level=2)
    store.upsert_heading(slug="ml", title="Machine Learning v2", section="AI", level=2)

    assert store.count() == 1
    # The latest title should win
    hits = store.query("Machine Learning v2", top_k=1)
    assert hits[0].slug == "ml"


def test_query_empty_collection_returns_empty(vector_engine: Any) -> None:
    """query() on an empty table returns []."""
    store = _store(vector_engine)
    assert store.query("anything") == []


def test_update_metadata_changes_section(vector_engine: Any) -> None:
    """update_metadata changes the section without re-embedding."""
    llm = _mock_llm()
    store = _store(vector_engine, llm)

    store.upsert_heading(slug="rl", title="Reinforcement Learning", section="Old", level=2)
    embed_call_count_before = llm.embed.call_count

    store.update_metadata(slug="rl", section="New Section")

    # No new embed call should have been made
    assert llm.embed.call_count == embed_call_count_before


def test_clear_resets_collection(vector_engine: Any) -> None:
    """clear() removes all entries and allows fresh insertion."""
    store = _store(vector_engine)

    for i in range(5):
        store.upsert_heading(slug=f"s{i}", title=f"T{i}", section="X", level=2)
    assert store.count() == 5

    store.clear()
    assert store.count() == 0

    store.upsert_heading(slug="new", title="New Page", section="X", level=2)
    assert store.count() == 1


def test_model_mismatch_raises_on_init(vector_engine: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """EmbeddingStore raises EmbeddingModelMismatchError if the stored model differs."""
    monkeypatch.setattr(settings, "embedding_model", "old-model")
    EmbeddingStore(llm_client=_mock_llm(), engine=vector_engine)  # records old-model

    monkeypatch.setattr(settings, "embedding_model", "new-model")
    with pytest.raises(EmbeddingModelMismatchError):
        EmbeddingStore(llm_client=_mock_llm(), engine=vector_engine)


def test_vector_failure_does_not_break_index(tmp_path: Path) -> None:
    """If EmbeddingStore.upsert_heading raises, IndexStorage.add_page still writes."""
    from llm_wiki.storage.index import IndexStorage

    index_path = tmp_path / "index.md"
    index_path.write_text("# Wiki Index\n\n## General\n\n")

    broken_store = MagicMock()
    broken_store.upsert_heading.side_effect = RuntimeError("vector store is down")

    storage = IndexStorage(index_path=index_path, embedding_store=broken_store)
    # Must NOT raise even though embedding store throws
    storage.add_page("test-page", "General")

    assert "[[test-page]]" in index_path.read_text()


def test_search_hit_is_frozen() -> None:
    """SearchHit is a frozen dataclass — immutable after construction."""
    hit = SearchHit(slug="x", title="X", section="General", similarity=0.9)
    with pytest.raises(Exception):
        hit.slug = "y"  # type: ignore[misc]


def test_usage_logged_after_embed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """embed() writes a record to usage.log with the correct model."""
    import openai

    from llm_wiki.config import settings as real_settings
    from llm_wiki.llm.client import LLMClient

    usage_log = tmp_path / "usage.log"
    # embed() reads settings via a local import, so patch the real singleton.
    monkeypatch.setattr(real_settings, "openai_api_key", "test-key")

    fake_response = MagicMock()
    fake_response.data = [MagicMock(embedding=[0.1] * 1536)]
    fake_response.usage.total_tokens = 5

    client = LLMClient.__new__(LLMClient)
    client._provider = "openai"
    client._usage_log_path = usage_log
    client._model = "gpt-5.4-mini"
    client._non_retryable = (Exception,)  # won't match real errors
    client._budget = MagicMock()  # embed() runs a budget check first

    sync_oa = MagicMock(spec=openai.OpenAI)
    sync_oa.embeddings.create.return_value = fake_response

    with patch("llm_wiki.llm.client.openai.OpenAI", return_value=sync_oa):
        result = client.embed(["hello world"])

    assert result == [[0.1] * 1536]
    assert usage_log.exists()
    lines = [json.loads(ln) for ln in usage_log.read_text().splitlines() if ln]
    assert lines[0]["agent_type"] == "embed"
    assert lines[0]["model"] == "text-embedding-3-small"
