"""pgvector-backed embedding store for wiki page headings.

Uses OpenAI text-embedding-3-small (via LLMClient.embed) to index all
headings from index.md.  Vector similarity search provides a fast pre-filter
for the Search Agent (LW-12) before the more expensive LLM re-rank step.

Storage: the ``heading_embeddings`` table in PostgreSQL (pgvector extension),
one row per heading. Distance metric: cosine (``<=>``), so
    similarity = 1.0 - distance
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import structlog
from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine

from llm_wiki.storage.metadata import (
    EmbeddingMeta,
    HeadingEmbedding,
    get_sync_engine,
)

logger = structlog.get_logger(__name__)

_SCOPE = "headings"


class EmbeddingError(RuntimeError):
    """Raised when an embedding operation fails after all retries."""


class EmbeddingModelMismatchError(EmbeddingError):
    """Raised when the stored vectors use a different model than config."""


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
    """Thin wrapper around PostgreSQL/pgvector + OpenAI embeddings.

    Stores one embedding per wiki heading (level-1 and level-2 only) in the
    ``heading_embeddings`` table. Metadata per row: slug, title, section, level,
    file_id, last_indexed_at.

    Design principles:
    - Synchronous: embedding calls use ``LLMClient.embed`` (sync); the store
      uses its own synchronous SQLAlchemy engine (the app's main engine is
      async, but these calls happen in sync contexts too — the Celery worker).
    - Idempotent: upsert operations are safe to call repeatedly (ON CONFLICT).
    - Isolated: failures do NOT propagate to the caller unless the embedding
      store is the primary operation. index.md (source of truth) writes happen
      before vector updates.
    """

    def __init__(
        self,
        llm_client: "LLMClient",  # noqa: F821 — avoids circular import at top level
        engine: Engine | None = None,
    ) -> None:
        """Initialise the store and validate the model/dim guard.

        Args:
            llm_client: Client used to call the OpenAI embeddings API.
            engine: Optional synchronous SQLAlchemy engine (injected in tests);
                defaults to the shared sync engine on ``DATABASE_URL``.

        Raises:
            EmbeddingModelMismatchError: If existing vectors were created with
                a different embedding model or dimension than the current config.
        """
        from llm_wiki.config import settings
        from llm_wiki.llm.client import LLMClient  # noqa: PLC0415

        self._llm: LLMClient = llm_client  # type: ignore[assignment]
        self._model: str = settings.embedding_model
        self._dim: int = settings.embedding_dimensions
        self._engine: Engine = engine or get_sync_engine()

        self._validate_meta()

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

        Raises:
            EmbeddingError: If the OpenAI API call fails after all retries.
        """
        try:
            vectors = self._llm.embed([title], file_id=file_id)
        except Exception as exc:
            raise EmbeddingError(f"embed failed for slug {slug!r}: {exc}") from exc

        row = {
            "slug": slug,
            "title": title,
            "section": section,
            "level": level,
            "file_id": file_id,
            "last_indexed_at": datetime.now(timezone.utc),
            "embedding": vectors[0],
        }
        with self._engine.begin() as conn:
            conn.execute(self._upsert_stmt([row]))
        logger.debug("embedding_upserted", slug=slug, section=section)

    def upsert_many(
        self,
        headings: "list[HeadingInfo]",  # noqa: F821
        file_id: str = "",
        slug_to_file_id: dict[str, str] | None = None,
    ) -> None:
        """Batch-upsert multiple headings.

        Processes in chunks of ``EMBEDDING_BATCH_SIZE`` (config) to stay
        within the OpenAI embeddings API input limit.
        """
        if not headings:
            return

        slug_map = slug_to_file_id or {}

        from llm_wiki.config import settings

        batch_size = settings.embedding_batch_size
        for i in range(0, len(headings), batch_size):
            chunk = headings[i : i + batch_size]
            titles = [h.title for h in chunk]
            try:
                vectors = self._llm.embed(titles, file_id=file_id)
            except Exception as exc:
                raise EmbeddingError(f"batch embed failed at offset {i}: {exc}") from exc

            now = datetime.now(timezone.utc)
            rows = [
                {
                    "slug": h.slug,
                    "title": h.title,
                    "section": h.section,
                    "level": h.level,
                    "file_id": slug_map.get(h.slug, file_id),
                    "last_indexed_at": now,
                    "embedding": vec,
                }
                for h, vec in zip(chunk, vectors, strict=True)
            ]
            with self._engine.begin() as conn:
                conn.execute(self._upsert_stmt(rows))
        logger.info("embedding_batch_upserted", count=len(headings))

    def backfill_file_id(self, slug: str, file_id: str) -> bool:
        """Set ``file_id`` on an existing heading without re-embedding."""
        try:
            with self._engine.begin() as conn:
                conn.execute(
                    update(HeadingEmbedding)
                    .where(HeadingEmbedding.slug == slug)
                    .values(file_id=file_id)
                )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("embedding_backfill_failed", slug=slug, error=str(exc))
            return False

    def update_metadata(self, slug: str, section: str) -> None:
        """Update only the *section* for an existing heading (no re-embed).

        Called by ``IndexStorage.move_page``.
        """
        try:
            with self._engine.begin() as conn:
                conn.execute(
                    update(HeadingEmbedding)
                    .where(HeadingEmbedding.slug == slug)
                    .values(section=section)
                )
            logger.debug("embedding_metadata_updated", slug=slug, section=section)
        except Exception as exc:  # noqa: BLE001
            logger.warning("embedding_metadata_update_failed", slug=slug, error=str(exc))

    def delete(self, slug: str) -> None:
        """Remove the embedding for *slug*. No-op if not present."""
        try:
            with self._engine.begin() as conn:
                conn.execute(
                    delete(HeadingEmbedding).where(HeadingEmbedding.slug == slug)
                )
            logger.debug("embedding_deleted", slug=slug)
        except Exception as exc:  # noqa: BLE001
            logger.warning("embedding_delete_failed", slug=slug, error=str(exc))

    def query(self, text: str, top_k: int = 20, file_id: str = "") -> list[SearchHit]:
        """Return the *top_k* most similar headings to *text*.

        Returns:
            SearchHit list sorted by descending similarity. Empty if the table
            is empty.

        Raises:
            EmbeddingError: If the OpenAI embeddings API call fails.
        """
        if self.count() == 0:
            return []

        try:
            vectors = self._llm.embed([text], file_id=file_id)
        except Exception as exc:
            raise EmbeddingError(f"embed failed during query: {exc}") from exc

        qvec = vectors[0]
        distance = HeadingEmbedding.embedding.cosine_distance(qvec)
        stmt = (
            select(
                HeadingEmbedding.slug,
                HeadingEmbedding.title,
                HeadingEmbedding.section,
                distance.label("distance"),
            )
            .order_by(distance)
            .limit(top_k)
        )
        with self._engine.connect() as conn:
            rows = conn.execute(stmt).all()

        hits = [
            SearchHit(
                slug=r.slug,
                title=r.title or r.slug,
                section=r.section or "",
                similarity=max(0.0, 1.0 - float(r.distance)),
            )
            for r in rows
        ]
        hits.sort(key=lambda h: h.similarity, reverse=True)
        return hits

    def count(self) -> int:
        """Return the number of headings currently indexed."""
        with self._engine.connect() as conn:
            return int(
                conn.execute(select(func.count()).select_from(HeadingEmbedding)).scalar_one()
            )

    def clear(self) -> None:
        """Delete ALL heading embeddings. Used by ``scripts/reindex.py``."""
        with self._engine.begin() as conn:
            conn.execute(delete(HeadingEmbedding))
        self._write_meta()
        logger.info("embedding_collection_cleared")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _upsert_stmt(self, rows: list[dict[str, object]]):  # type: ignore[no-untyped-def]
        """Build an INSERT ... ON CONFLICT(slug) DO UPDATE statement."""
        ins = pg_insert(HeadingEmbedding).values(rows)
        return ins.on_conflict_do_update(
            index_elements=[HeadingEmbedding.slug],
            set_={
                "title": ins.excluded.title,
                "section": ins.excluded.section,
                "level": ins.excluded.level,
                "file_id": ins.excluded.file_id,
                "last_indexed_at": ins.excluded.last_indexed_at,
                "embedding": ins.excluded.embedding,
            },
        )

    def _validate_meta(self) -> None:
        """Guard against a model/dim change without a reindex.

        Mirrors the old ChromaDB collection-metadata check: if a prior build
        used a different embedding model or dimension, refuse to run so stale
        vectors are never mixed with new ones.
        """
        with self._engine.begin() as conn:
            stored = conn.execute(
                select(EmbeddingMeta.model, EmbeddingMeta.dim).where(
                    EmbeddingMeta.scope == _SCOPE
                )
            ).first()
            if stored is None:
                conn.execute(self._meta_upsert_stmt())
                return
            if stored.model != self._model:
                raise EmbeddingModelMismatchError(
                    f"heading_embeddings were built with model {stored.model!r}, "
                    f"but config says {self._model!r}. "
                    "Run: docker compose exec api uv run python scripts/reindex.py"
                )
            if stored.dim != self._dim:
                raise EmbeddingModelMismatchError(
                    f"heading_embeddings dimension {stored.dim} != config {self._dim}. "
                    "Run: docker compose exec api uv run python scripts/reindex.py"
                )

    def _write_meta(self) -> None:
        """Record the current model/dim for the headings scope."""
        with self._engine.begin() as conn:
            conn.execute(self._meta_upsert_stmt())

    def _meta_upsert_stmt(self):  # type: ignore[no-untyped-def]
        ins = pg_insert(EmbeddingMeta).values(
            scope=_SCOPE, model=self._model, dim=self._dim
        )
        return ins.on_conflict_do_update(
            index_elements=[EmbeddingMeta.scope],
            set_={"model": ins.excluded.model, "dim": ins.excluded.dim},
        )


@dataclass
class HeadingInfo:
    """Lightweight container used by upsert_many / reindex.py."""

    slug: str
    title: str
    section: str
    level: int
