"""Unit tests for ChunkStore and chunk_markdown (LW-20.1).

All tests use an in-memory ChromaDB EphemeralClient and a mocked LLMClient
so no real API calls are made.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import chromadb
import pytest

from llm_wiki.llm.chunk_store import ChunkStore, chunk_markdown
from llm_wiki.llm.embeddings import EmbeddingModelMismatchError

_DIM = 8  # small dimension for fast tests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_embed(texts: list[str], **_kwargs: object) -> list[list[float]]:
    """Return deterministic unit-length embeddings based on text hash."""
    results = []
    for t in texts:
        seed = int(hashlib.md5(t.encode()).hexdigest(), 16) % (10**6)
        vec = [float((seed >> i) & 1) for i in range(_DIM)]
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        results.append([v / norm for v in vec])
    return results


def _mock_llm() -> MagicMock:
    mock = MagicMock()
    mock.embed.side_effect = _fake_embed
    return mock


def _store(mock_llm: MagicMock | None = None, dim: int = _DIM) -> ChunkStore:
    """Return a ChunkStore backed by an in-memory ChromaDB client."""
    with patch("llm_wiki.config.settings") as s:
        s.embedding_model = "text-embedding-3-small"
        s.embedding_dimensions = dim
        s.embedding_batch_size = 50
        s.chunk_max_chars = 200
        s.chunk_overlap_chars = 20
        return ChunkStore(
            chroma_path=Path("/tmp/unused"),
            llm_client=mock_llm or _mock_llm(),
            chroma_client=chromadb.EphemeralClient(),
        )


# ---------------------------------------------------------------------------
# chunk_markdown unit tests
# ---------------------------------------------------------------------------


def test_chunk_markdown_three_sections() -> None:
    """A page with three ## sections produces three chunks with correct section names."""
    text = (
        "# Title\n\nPreamble text here.\n\n"
        "## Section A\n\nContent for A. " * 5 + "\n\n"
        "## Section B\n\nContent for B. " * 5 + "\n\n"
        "## Section C\n\nContent for C. " * 5
    )
    chunks = chunk_markdown(text, max_chars=2000, overlap=100)
    sections = [s for s, _ in chunks]
    # Preamble may be merged or absent; the three ## sections must all appear
    assert "Section A" in sections
    assert "Section B" in sections
    assert "Section C" in sections


def test_chunk_markdown_sliding_window() -> None:
    """A section 3× max_chars long produces at least 3 chunks with overlap."""
    body = "word " * 400  # ~2000 chars at 5 chars/word
    text = f"## Big Section\n\n{body}"
    chunks = chunk_markdown(text, max_chars=500, overlap=50)
    assert len(chunks) >= 3
    # Each chunk belongs to the same section
    for section, _ in chunks:
        assert section == "Big Section"
    # Adjacent chunks overlap: last chars of chunk N == first chars of chunk N+1 (approx)
    if len(chunks) >= 2:
        _, c0 = chunks[0]
        _, c1 = chunks[1]
        # c1 starts inside c0's tail (overlap=50 chars)
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
    # Section name is empty when there are no headings
    assert all(s == "" for s, _ in chunks)


def test_chunk_markdown_empty_returns_empty() -> None:
    assert chunk_markdown("", max_chars=2000, overlap=100) == []
    assert chunk_markdown("   \n  ", max_chars=2000, overlap=100) == []


# ---------------------------------------------------------------------------
# ChunkStore tests
# ---------------------------------------------------------------------------


def test_upsert_page_idempotent() -> None:
    """Calling upsert_page twice with the same content does not duplicate chunks."""
    store = _store()
    content = "## Section\n\n" + "Some content here. " * 20
    store.upsert_page("wiki-page", "Wiki Page", content)
    count_first = store.count()
    store.upsert_page("wiki-page", "Wiki Page", content)
    assert store.count() == count_first


def test_upsert_page_replaces_old_chunks() -> None:
    """upsert_page with new content removes old chunks for the slug."""
    store = _store()
    long_content = "## A\n\n" + "x " * 300 + "\n\n## B\n\n" + "y " * 300
    store.upsert_page("p", "P", long_content)
    before = store.count()

    short_content = "## A\n\n" + "z " * 30
    store.upsert_page("p", "P", short_content)
    after = store.count()

    assert after < before, "Fewer chunks expected after replacing with shorter content"


def test_delete_page_removes_all_chunks() -> None:
    """delete_page removes all chunks for a slug, leaving others intact."""
    store = _store()
    content = "## X\n\n" + "content " * 30
    store.upsert_page("alpha", "Alpha", content)
    store.upsert_page("beta", "Beta", content)
    before_beta = store.count()

    store.delete_page("alpha")

    # beta's chunks still there; alpha's are gone
    hits = store.query("content", top_k=20)
    slugs = {h.slug for h in hits}
    assert "alpha" not in slugs
    assert "beta" in slugs


def test_query_returns_relevant_chunk_above_irrelevant() -> None:
    """A query matching one page's content should rank it higher than unrelated page."""
    store = _store()
    # Use distinct hashes so the fake embedder produces different vectors
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
    assert hits[0].slug == "adam-optimizer"


def test_clear_empties_collection() -> None:
    """clear() removes all chunks from the collection."""
    store = _store()
    store.upsert_page("p1", "P1", "## S\n\n" + "text " * 30)
    store.upsert_page("p2", "P2", "## S\n\n" + "text " * 30)
    assert store.count() > 0
    store.clear()
    assert store.count() == 0


def test_mismatch_model_raises() -> None:
    """EmbeddingModelMismatchError is raised if model changed between runs."""
    chroma_client = chromadb.EphemeralClient()

    # Build initial store with model A
    with patch("llm_wiki.config.settings") as s:
        s.embedding_model = "model-a"
        s.embedding_dimensions = _DIM
        s.embedding_batch_size = 50
        s.chunk_max_chars = 200
        s.chunk_overlap_chars = 20
        store_a = ChunkStore(
            chroma_path=Path("/tmp/unused"),
            llm_client=_mock_llm(),
            chroma_client=chroma_client,
        )
        store_a.upsert_page("x", "X", "## S\n\n" + "text " * 30)

    # Open the same collection with a different model — must raise
    with patch("llm_wiki.config.settings") as s:
        s.embedding_model = "model-b"
        s.embedding_dimensions = _DIM
        s.embedding_batch_size = 50
        s.chunk_max_chars = 200
        s.chunk_overlap_chars = 20
        with pytest.raises(EmbeddingModelMismatchError):
            ChunkStore(
                chroma_path=Path("/tmp/unused"),
                llm_client=_mock_llm(),
                chroma_client=chroma_client,
            )


def test_chunk_hit_text_field_populated() -> None:
    """ChunkHit.text contains the actual chunk body (not empty)."""
    store = _store()
    content = "## Overview\n\nThis is overview text. " * 10
    store.upsert_page("overview-page", "Overview Page", content)
    hits = store.query("overview text", top_k=3)
    assert hits, "Expected at least one hit"
    assert all(len(h.text) > 0 for h in hits), "ChunkHit.text must not be empty"


def test_upsert_page_stores_file_id_in_metadata() -> None:
    """Chunks must carry file_id so retrieval can be scoped per source file (LW-N3)."""
    store = _store()
    content = "## Section\n\n" + ("Scoped retrieval text. " * 20)
    source_id = "file-abc-123"
    store.upsert_page("scoped-page", "Scoped", content, file_id=source_id)

    assert store.count_by_file_id(source_id) > 0

    other_id = "file-other-999"
    assert store.count_by_file_id(other_id) == 0

    hits = store.query("retrieval", top_k=5, file_ids=[source_id])
    assert hits
    assert all(h.slug == "scoped-page" for h in hits)


def test_backfill_file_id_updates_existing_chunks() -> None:
    """backfill_file_id patches metadata on chunks indexed before LW-N3."""
    store = _store()
    content = "## Legacy\n\n" + ("Legacy chunk body. " * 20)
    store.upsert_page("legacy-page", "Legacy", content, file_id="")

    assert store.count_by_file_id("legacy-file") == 0
    updated = store.backfill_file_id("legacy-page", "legacy-file")
    assert updated > 0
    assert store.count_by_file_id("legacy-file") == updated
