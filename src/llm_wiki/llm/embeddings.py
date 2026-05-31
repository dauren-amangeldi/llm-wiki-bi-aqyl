"""ChromaDB-backed embedding store for wiki page headings.

Uses OpenAI text-embedding-3-small (via LLMClient.embed) to index all
headings from index.md.  Vector similarity search provides a fast pre-filter
for the Search Agent (LW-12) before the more expensive LLM re-rank step.

Distance metric: cosine.  ChromaDB stores cosine distance in [0, 2], so
    similarity = 1.0 - distance
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import chromadb
import structlog

logger = structlog.get_logger(__name__)

_COLLECTION_NAME = "headings"


class EmbeddingError(RuntimeError):
    """Raised when an embedding operation fails after all retries."""


class EmbeddingModelMismatchError(EmbeddingError):
    """Raised when the stored collection uses a different model than config."""


@dataclass(frozen=True)
class SearchHit:
    """A single wiki heading matched by vector similarity (and optionally LLM-reranked).

    ``similarity`` is the cosine similarity from the embedding pre-filter.
    ``rerank_score`` is set by the LLM reranker (None if LLM was skipped / timed out).
    ``reason`` is the one-sentence LLM justification (None when reranker was skipped).
    """

    slug: str
    title: str
    section: str
    similarity: float
    rerank_score: float | None = field(default=None)
    reason: str | None = field(default=None)


class EmbeddingStore:
    """Thin wrapper around ChromaDB + OpenAI embeddings.

    Stores one embedding per wiki heading (level-1 and level-2 only).
    Metadata stored per entry: slug, title, section, level, last_indexed_at.

    Design principles:
    - Synchronous: ChromaDB's Python client is synchronous; embedding calls
      use LLMClient.embed() (also sync) to keep the interface simple.
    - Idempotent: upsert operations are safe to call repeatedly.
    - Isolated: failures do NOT propagate to the caller unless embedding_store
      is the primary operation.  index.md (source of truth) writes happen
      before Chroma updates.
    """

    def __init__(
        self,
        chroma_path: Path,
        llm_client: "LLMClient",  # noqa: F821 — avoids circular import at top level
        chroma_client: chromadb.ClientAPI | None = None,
    ) -> None:
        """Initialise and validate the ChromaDB collection.

        Args:
            chroma_path: Directory for ChromaDB persistence (created if absent).
            llm_client: Client used to call the OpenAI embeddings API.
            chroma_client: Optional pre-constructed ChromaDB client (used in tests
                to inject an in-memory ``EphemeralClient``).

        Raises:
            EmbeddingModelMismatchError: If an existing collection was created with
                a different embedding model or dimension than the current config.
        """
        from llm_wiki.config import settings

        # Import here to avoid polluting the top-level namespace unnecessarily.
        from llm_wiki.llm.client import LLMClient  # noqa: PLC0415

        self._llm: LLMClient = llm_client  # type: ignore[assignment]
        self._model: str = settings.embedding_model
        self._dim: int = settings.embedding_dimensions

        if chroma_client is not None:
            self._chroma = chroma_client
        else:
            chroma_path.mkdir(parents=True, exist_ok=True)
            self._chroma = chromadb.PersistentClient(path=str(chroma_path))

        self._col = self._get_or_create_collection()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upsert_heading(
        self,
        slug: str,
        title: str,
        section: str,
        level: int,
        file_id: str = "",
    ) -> None:
        """Create or replace the embedding for a single heading.

        Idempotent: calling with the same *slug* replaces the existing entry.

        Args:
            slug: Unique page identifier (e.g. ``"transformers"``).
            title: Human-readable page title used to compute the embedding.
            section: index.md section containing this heading.
            level: Markdown heading level (1 = ``#``, 2 = ``##``, …).
            file_id: Correlation ID passed to usage tracking.

        Raises:
            EmbeddingError: If the OpenAI API call fails after all retries.
        """
        try:
            vectors = self._llm.embed([title], file_id=file_id)
        except Exception as exc:
            raise EmbeddingError(f"embed failed for slug {slug!r}: {exc}") from exc

        self._col.upsert(
            ids=[slug],
            embeddings=[vectors[0]],
            metadatas=[
                {
                    "slug": slug,
                    "title": title,
                    "section": section,
                    "level": level,
                    "last_indexed_at": datetime.now(timezone.utc).isoformat(),
                }
            ],
            documents=[title],
        )
        logger.debug("embedding_upserted", slug=slug, section=section)

    def upsert_many(
        self,
        headings: "list[HeadingInfo]",  # noqa: F821
        file_id: str = "",
    ) -> None:
        """Batch-upsert multiple headings.

        Processes in chunks of ``EMBEDDING_BATCH_SIZE`` (config) to stay
        within the OpenAI embeddings API input limit.

        Args:
            headings: List of ``HeadingInfo`` objects with slug/title/section/level.
            file_id: Correlation ID for usage tracking.

        Raises:
            EmbeddingError: If any API call fails after all retries.
        """
        if not headings:
            return

        from llm_wiki.config import settings

        batch_size = settings.embedding_batch_size
        for i in range(0, len(headings), batch_size):
            chunk = headings[i : i + batch_size]
            titles = [h.title for h in chunk]
            try:
                vectors = self._llm.embed(titles, file_id=file_id)
            except Exception as exc:
                raise EmbeddingError(f"batch embed failed at offset {i}: {exc}") from exc

            self._col.upsert(
                ids=[h.slug for h in chunk],
                embeddings=vectors,
                metadatas=[
                    {
                        "slug": h.slug,
                        "title": h.title,
                        "section": h.section,
                        "level": h.level,
                        "last_indexed_at": datetime.now(timezone.utc).isoformat(),
                    }
                    for h in chunk
                ],
                documents=[h.title for h in chunk],
            )
        logger.info("embedding_batch_upserted", count=len(headings))

    def update_metadata(self, slug: str, section: str) -> None:
        """Update only the *section* metadata for an existing heading.

        Called by ``IndexStorage.move_page``.  The embedding vector is NOT
        recomputed since the heading title has not changed.

        Args:
            slug: Page slug whose metadata to update.
            section: New section name.
        """
        try:
            self._col.update(
                ids=[slug],
                metadatas=[{"section": section}],
            )
            logger.debug("embedding_metadata_updated", slug=slug, section=section)
        except Exception as exc:  # noqa: BLE001
            # Non-fatal: log and continue (Chroma is a cache, not source of truth)
            logger.warning("embedding_metadata_update_failed", slug=slug, error=str(exc))

    def delete(self, slug: str) -> None:
        """Remove the embedding for *slug*.  No-op if not present.

        Args:
            slug: Page slug to remove.
        """
        try:
            self._col.delete(ids=[slug])
            logger.debug("embedding_deleted", slug=slug)
        except Exception as exc:  # noqa: BLE001
            logger.warning("embedding_delete_failed", slug=slug, error=str(exc))

    def query(self, text: str, top_k: int = 20, file_id: str = "") -> list[SearchHit]:
        """Return the *top_k* most similar headings to *text*.

        Args:
            text: Query string (e.g. a document summary).
            top_k: Maximum number of results to return.
            file_id: Correlation ID for embedding usage tracking.

        Returns:
            SearchHit list sorted by descending similarity.  Empty if the
            collection is empty or the embedding API call fails.

        Raises:
            EmbeddingError: If the OpenAI embeddings API call fails.
        """
        current_count = self.count()
        if current_count == 0:
            return []

        n = min(top_k, current_count)
        try:
            vectors = self._llm.embed([text], file_id=file_id)
        except Exception as exc:
            raise EmbeddingError(f"embed failed during query: {exc}") from exc

        results = self._col.query(
            query_embeddings=[vectors[0]],
            n_results=n,
            include=["metadatas", "distances"],  # type: ignore[list-item]
        )

        hits: list[SearchHit] = []
        ids: list[str] = results.get("ids", [[]])[0]  # type: ignore[index]
        distances: list[float] = results.get("distances", [[]])[0]  # type: ignore[index]
        metas: list[dict[str, object]] = results.get("metadatas", [[]])[0]  # type: ignore[index]

        for slug, dist, meta in zip(ids, distances, metas, strict=True):
            similarity = max(0.0, 1.0 - float(dist))
            hits.append(
                SearchHit(
                    slug=slug,
                    title=str(meta.get("title", slug)),
                    section=str(meta.get("section", "")),
                    similarity=similarity,
                )
            )

        hits.sort(key=lambda h: h.similarity, reverse=True)
        return hits

    def count(self) -> int:
        """Return the number of headings currently indexed.

        Returns:
            Non-negative integer.
        """
        return self._col.count()  # type: ignore[no-any-return]

    def clear(self) -> None:
        """Delete ALL entries from the collection.

        Used by ``scripts/reindex.py`` before a full rebuild.
        """
        self._chroma.delete_collection(_COLLECTION_NAME)
        self._col = self._get_or_create_collection(fresh=True)
        logger.info("embedding_collection_cleared")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_or_create_collection(
        self, fresh: bool = False
    ) -> chromadb.Collection:
        """Return (or create) the headings collection, validating model metadata.

        Args:
            fresh: If True, do not validate existing metadata (used after clear()).

        Returns:
            ChromaDB Collection object.

        Raises:
            EmbeddingModelMismatchError: If the stored model/dim differs from config.
        """
        wanted_meta = {
            "hnsw:space": "cosine",
            "embedding_model": self._model,
            "embedding_dim": str(self._dim),
        }

        # Suppress noisy ChromaDB telemetry logs
        logging.getLogger("chromadb").setLevel(logging.WARNING)

        col = self._chroma.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata=wanted_meta,
        )

        if not fresh:
            stored = col.metadata or {}
            stored_model = stored.get("embedding_model")
            stored_dim = stored.get("embedding_dim")
            if stored_model and stored_model != self._model:
                raise EmbeddingModelMismatchError(
                    f"ChromaDB collection was built with model {stored_model!r}, "
                    f"but config says {self._model!r}. "
                    "Run: docker compose exec api uv run python scripts/reindex.py"
                )
            if stored_dim and stored_dim != str(self._dim):
                raise EmbeddingModelMismatchError(
                    f"ChromaDB collection dimension {stored_dim} != config {self._dim}. "
                    "Run: docker compose exec api uv run python scripts/reindex.py"
                )

        return col


@dataclass
class HeadingInfo:
    """Lightweight container used by upsert_many / reindex.py."""

    slug: str
    title: str
    section: str
    level: int
