"""Thin wrapper around ChunkStore.upsert_page for use in the ingestion pipeline.

Provides a single, exception-safe call site so that ``pipeline.py`` does not
scatter try/except blocks around chunk-indexing logic.  All failures are
logged but never re-raised — wiki/*.md is the source of truth; the chunk
index is a derivative cache.
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)


def sync_chunks_for_page(
    chunk_store: "ChunkStore | None",  # noqa: F821
    slug: str,
    title: str,
    content: str,
    file_id: str = "",
) -> None:
    """Update the chunk index for one wiki page after it was saved to disk.

    No-op when *chunk_store* is ``None`` (feature flag / test isolation).
    Swallows all exceptions so a Chroma failure never aborts the ingestion
    pipeline.

    Args:
        chunk_store: The ``ChunkStore`` instance to update, or ``None``.
        slug: Wiki page slug (e.g. ``"transformers"``).
        title: Human-readable page title.
        content: Full markdown content of the page that was just saved.
        file_id: Correlation ID for log / usage tracking.
    """
    if chunk_store is None:
        return
    try:
        chunk_store.upsert_page(slug=slug, title=title, content=content, file_id=file_id)
        logger.debug("chunk_sync_ok", slug=slug, file_id=file_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("chunk_sync_failed", slug=slug, file_id=file_id, error=str(exc))
