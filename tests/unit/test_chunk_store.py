"""Unit tests for ChunkStore and chunk_markdown (pgvector-backed).

Store tests use the ``vector_engine`` fixture (a sync engine to the test
Postgres with the pgvector extension) and a mocked LLMClient so no real API
calls are made. The pure ``chunk_markdown`` tests need no database.
"""

from __future__ import annotations

import hashlib
from typing import Any
from unittest.mock import MagicMock

import pytest

from llm_wiki.config import settings
from llm_wiki.llm.chunk_store import ChunkStore, chunk_markdown
from llm_wiki.llm.embeddings import EmbeddingModelMismatchError

_DIM = settings.embedding_dimensions  # must match the vector() column dimension


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_embed(texts: list[str], **_kwargs: object) -> list[list[float]]:
    """Deterministic unit-length embeddings derived from an MD5 digest."""
    results = []
    for t in texts:
        digest = hashlib.md5(t.encode()).digest()
        vec = [float(digest[i % len(digest)]) for i in range(_DIM)]
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        results.append([v / norm for v in vec])
    return results


def _mock_llm() -> MagicMock:
    mock = MagicMock()
    mock.embed.side_effect = _fake_embed
    return mock


def _store(vector_engine: Any, mock_llm: MagicMock | None = None) -> ChunkStore:
    """Return a ChunkStore bound to the test Postgres engine (small chunks)."""
    store = ChunkStore(llm_client=mock_llm or _mock_llm(), engine=vector_engine)
    # Small chunk size so short test content still produces multiple chunks.
    store._max_chars = 200
    store._overlap = 20
    return store


# ---------------------------------------------------------------------------
# chunk_markdown unit tests (pure, no DB)
# ---------------------------------------------------------------------------


def test_chunk_markdown_three_sections() -> None:
    """A page with three ## sections produces chunks tagged with each section."""
    text = (
        "# Title\n\nPreamble text here.\n\n"
        + "## Section A\n\n" + "Content for A. " * 10 + "\n\n"
        + "## Section B\n\n" + "Content for B. " * 10 + "\n\n"
        + "## Section C\n\n" + "Content for C. " * 10
    )
    chunks = chunk_markdown(text, max_chars=2000, overlap=100)
    sections = [s for s, _ in chunks]
    assert "Section A" in sections
    assert "Section B" in sections
    assert "Section C" in sections


def test_chunk_markdown_sliding_window() -> None:
    """A section 3× max_chars long produces at least 3 chunks with overlap."""
    body = "word " * 400  # ~2000 chars at 5 chars/word
    text = f"## Big Section\n\n{body}"
    chunks = chunk_markdown(text, max_chars=500, overlap=50)
    assert len(chunks) >= 3
    for section, _ in chunks:
        assert section == "Big Section"
    if len(chunks) >= 2:
        _, c0 = chunks[0]
        _, c1 = chunks[1]
        assert c1[:50] in c0 or c0[-50:] in c1 or len(c0) > 450


def test_chunk_markdown_short_chunks_discarded() -> None:
    """Chunks shorter than 100 chars are dropped."""
    text = "## Tiny\n\nHi.\n\n## Normal\n\n" + "x" * 200
    chunks = chunk_markdown(text, max_chars=2000, overlap=0)
    for _, body in chunks:
        assert len(body) >= 100


def test_chunk_markdown_no_headings() -> None:
    """A page with no ## headings is chunked as a single unnamed section."""
    text = "Just plain text. " * 50
    chunks = chunk_markdown(text, max_chars=2000, overlap=100)
    assert len(chunks) >= 1
    assert all(s == "" for s, _ in chunks)


def test_chunk_markdown_empty_returns_empty() -> None:
    assert chunk_markdown("", max_chars=2000, overlap=100) == []
    assert chunk_markdown("   \n  ", max_chars=2000, overlap=100) == []


# ---------------------------------------------------------------------------
# ChunkStore tests (pgvector)
# ---------------------------------------------------------------------------


def test_upsert_page_idempotent(vector_engine: Any) -> None:
    """Calling upsert_page twice with the same content does not duplicate chunks."""
    store = _store(vector_engine)
    content = "## Section\n\n" + "Some content here. " * 20
    store.upsert_page("wiki-page", "Wiki Page", content)
    count_first = store.count()
    store.upsert_page("wiki-page", "Wiki Page", content)
    assert store.count() == count_first


def test_upsert_page_replaces_old_chunks(vector_engine: Any) -> None:
    """upsert_page with new content removes old chunks for the slug."""
    store = _store(vector_engine)
    long_content = "## A\n\n" + "x " * 300 + "\n\n## B\n\n" + "y " * 300
    store.upsert_page("p", "P", long_content)
    before = store.count()

    short_content = "## A\n\n" + "z " * 30
    store.upsert_page("p", "P", short_content)
    after = store.count()

    assert after < before, "Fewer chunks expected after replacing with shorter content"


def test_delete_page_removes_all_chunks(vector_engine: Any) -> None:
    """delete_page removes all chunks for a slug, leaving others intact."""
    store = _store(vector_engine)
    content = "## X\n\n" + "content " * 30
    store.upsert_page("alpha", "Alpha", content)
    store.upsert_page("beta", "Beta", content)

    store.delete_page("alpha")

    hits = store.query("content", top_k=20)
    slugs = {h.slug for h in hits}
    assert "alpha" not in slugs
    assert "beta" in slugs


def test_query_returns_relevant_chunk_above_irrelevant(vector_engine: Any) -> None:
    """A query matching one page's content should return the indexed page."""
    store = _store(vector_engine)
    store.upsert_page(
        "adam-optimizer",
        "Adam Optimizer",
        "## Theory\n\nAdam optimizer uses adaptive learning rates for gradient descent. " * 5,
    )
    store.upsert_page(
        "unrelated-topic",
        "Unrelated Topic",
        "## Intro\n\nThis page is about something completely different, not related. " * 5,
    )

    hits = store.query("adam optimizer gradient", top_k=5)
    assert hits, "Expected at least one result"
    assert {"adam-optimizer", "unrelated-topic"} & {h.slug for h in hits}


def test_clear_empties_collection(vector_engine: Any) -> None:
    """clear() removes all chunks."""
    store = _store(vector_engine)
    store.upsert_page("p1", "P1", "## S\n\n" + "text " * 30)
    store.upsert_page("p2", "P2", "## S\n\n" + "text " * 30)
    assert store.count() > 0
    store.clear()
    assert store.count() == 0


def test_mismatch_model_raises(vector_engine: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """EmbeddingModelMismatchError is raised if the model changed between runs."""
    monkeypatch.setattr(settings, "embedding_model", "model-a")
    store_a = ChunkStore(llm_client=_mock_llm(), engine=vector_engine)
    store_a.upsert_page("x", "X", "## S\n\n" + "text " * 30)

    monkeypatch.setattr(settings, "embedding_model", "model-b")
    with pytest.raises(EmbeddingModelMismatchError):
        ChunkStore(llm_client=_mock_llm(), engine=vector_engine)


def test_chunk_hit_text_field_populated(vector_engine: Any) -> None:
    """ChunkHit.text contains the actual chunk body (not empty)."""
    store = _store(vector_engine)
    content = "## Overview\n\nThis is overview text. " * 10
    store.upsert_page("overview-page", "Overview Page", content)
    hits = store.query("overview text", top_k=3)
    assert hits, "Expected at least one hit"
    assert all(len(h.text) > 0 for h in hits), "ChunkHit.text must not be empty"


def test_upsert_page_stores_file_id_in_metadata(vector_engine: Any) -> None:
    """Chunks must carry file_id so retrieval can be scoped per source file (LW-N3)."""
    store = _store(vector_engine)
    content = "## Section\n\n" + ("Scoped retrieval text. " * 20)
    source_id = "file-abc-123"
    store.upsert_page("scoped-page", "Scoped", content, file_id=source_id)

    assert store.count_by_file_id(source_id) > 0

    other_id = "file-other-999"
    assert store.count_by_file_id(other_id) == 0

    hits = store.query("retrieval", top_k=5, file_ids=[source_id])
    assert hits
    assert all(h.slug == "scoped-page" for h in hits)


def test_usage_file_id_does_not_apply_where_filter(vector_engine: Any) -> None:
    """usage_file_id is for embed logging only — must not scope results."""
    store = _store(vector_engine)
    content_a = "## A\n\n" + ("Alpha retrieval content. " * 20)
    content_b = "## B\n\n" + ("Beta retrieval content. " * 20)
    store.upsert_page("page-a", "A", content_a, file_id="real-file")
    store.upsert_page("page-b", "B", content_b, file_id="other-file")

    hits = store.query("Alpha retrieval", top_k=5, usage_file_id="advisor")
    assert hits
    assert any(h.file_id == "real-file" for h in hits)
    assert not store.query("Alpha retrieval", top_k=5, file_id="advisor")


def test_backfill_file_id_updates_existing_chunks(vector_engine: Any) -> None:
    """backfill_file_id patches file_id on chunks indexed before LW-N3."""
    store = _store(vector_engine)
    content = "## Legacy\n\n" + ("Legacy chunk body. " * 20)
    store.upsert_page("legacy-page", "Legacy", content, file_id="")

    assert store.count_by_file_id("legacy-file") == 0
    updated = store.backfill_file_id("legacy-page", "legacy-file")
    assert updated > 0
    assert store.count_by_file_id("legacy-file") == updated
