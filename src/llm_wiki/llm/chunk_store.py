"""pgvector-backed chunk store for wiki page *bodies* (LW-20.1).

Why a second table instead of reusing ``heading_embeddings``?
------------------------------------------------------------
``EmbeddingStore`` (``heading_embeddings``) stores one embedding per page
**title**. It was designed for the *SearchAgent*: "given a new document,
which existing wiki pages should be updated?"  That is a page-level
classification task that works perfectly with title embeddings.

``ChunkStore`` (``chunk_embeddings``) stores one embedding per ~500-token
**text fragment** of a page body.  It is designed for the *AnswerAgent*:
"given a user question, which paragraph actually contains the answer?"
That is a passage-level retrieval task where title embeddings have poor
recall — a page titled "Training Runs" may contain the only mention of
"Adam optimizer" in the whole wiki, but its title embedding will never be
the nearest neighbour for the query "what optimizer do we use?".

The two tasks require different granularity, different top-k thresholds,
and different ID namespaces, so they live in separate tables.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import structlog
from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine

from llm_wiki.llm.embeddings import EmbeddingModelMismatchError
from llm_wiki.storage.metadata import (
    ChunkEmbedding,
    EmbeddingMeta,
    get_sync_engine,
)

logger = structlog.get_logger(__name__)

_SCOPE = "chunks"
_MIN_CHUNK_CHARS = 100  # discard chunks shorter than this (noise)


@dataclass(frozen=True)
class ChunkHit:
    """A single chunk returned by a similarity search over page bodies."""

    slug: str
    title: str    # page title (for UI / citation)
    section: str  # markdown section heading, empty if page has no ## headings
    chunk_idx: int
    text: str     # the actual chunk body (used directly as LLM context)
    similarity: float
    file_id: str = ""  # source document (case) ID, empty when not backfilled


# ---------------------------------------------------------------------------
# Pure chunking function
# ---------------------------------------------------------------------------


def chunk_markdown(
    text: str,
    max_chars: int = 2000,
    overlap: int = 200,
) -> list[tuple[str, str]]:
    """Split *text* into overlapping chunks, honouring markdown section boundaries.

    Algorithm:
    1. Split the document at ``##`` / ``###`` headings using a regex.  Each
       section becomes a candidate chunk labelled with its heading text.
    2. Sections longer than *max_chars* are further split with a sliding
       window of size *max_chars* and stride *max_chars - overlap*.
    3. Chunks shorter than ``_MIN_CHUNK_CHARS`` characters are discarded.

    Returns:
        List of ``(section_name, chunk_text)`` pairs.  ``section_name`` is the
        heading text of the section the chunk belongs to, or ``""`` for the
        preamble before the first heading.
    """
    if not text.strip():
        return []

    # Split at ## or ### headings (not #, which is usually the page title)
    _HEADING_RE = re.compile(r"^(#{2,3})\s+(.+)$", re.MULTILINE)

    sections: list[tuple[str, str]] = []  # (section_name, body_text)
    prev_end = 0
    prev_name = ""  # preamble has no section name

    for m in _HEADING_RE.finditer(text):
        # Everything between the previous heading and this one
        body = text[prev_end : m.start()].strip()
        if body:
            sections.append((prev_name, body))
        prev_name = m.group(2).strip()
        prev_end = m.end()

    # Trailing section after the last heading
    tail = text[prev_end:].strip()
    if tail:
        sections.append((prev_name, tail))

    # If no ## headings at all, treat the whole text as one unnamed section
    if not sections:
        sections = [("", text.strip())]

    # Apply sliding window to sections that are too long
    result: list[tuple[str, str]] = []
    stride = max(1, max_chars - overlap)

    for section_name, body in sections:
        if len(body) <= max_chars:
            chunk = body.strip()
            if len(chunk) >= _MIN_CHUNK_CHARS:
                result.append((section_name, chunk))
        else:
            start = 0
            while start < len(body):
                chunk = body[start : start + max_chars].strip()
                if len(chunk) >= _MIN_CHUNK_CHARS:
                    result.append((section_name, chunk))
                start += stride

    return result


# ---------------------------------------------------------------------------
# ChunkStore
# ---------------------------------------------------------------------------


class ChunkStore:
    """pgvector-backed chunk index for wiki page bodies.

    Stores one embedding per text chunk (≈500 tokens) in ``chunk_embeddings``
    so that the ``AnswerAgent`` can retrieve the specific paragraphs that
    contain the answer to a user question, rather than just the closest page
    title.

    Design principles mirror ``EmbeddingStore``:
    - Synchronous: embedding calls are sync; the store uses a sync engine.
    - Idempotent: ``upsert_page`` deletes old chunks before inserting new ones.
    - Isolated: failures are logged and NOT propagated (the wiki page is the
      source of truth; this index is a cache).
    """

    def __init__(
        self,
        llm_client: "LLMClient",  # noqa: F821
        engine: Engine | None = None,
    ) -> None:
        """Initialise the store and validate the model/dim guard.

        Args:
            llm_client: Used to call the embeddings API.
            engine: Optional synchronous SQLAlchemy engine (injected in tests);
                defaults to the shared sync engine on ``DATABASE_URL``.

        Raises:
            EmbeddingModelMismatchError: If existing vectors were built with a
                different embedding model or dimension.
        """
        from llm_wiki.config import settings
        from llm_wiki.llm.client import LLMClient  # noqa: PLC0415

        self._llm: LLMClient = llm_client  # type: ignore[assignment]
        self._model: str = settings.embedding_model
        self._dim: int = settings.embedding_dimensions
        self._max_chars: int = settings.chunk_max_chars
        self._overlap: int = settings.chunk_overlap_chars
        self._engine: Engine = engine or get_sync_engine()

        self._validate_meta()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upsert_page(
        self,
        slug: str,
        title: str,
        content: str,
        file_id: str = "",
    ) -> None:
        """Replace all chunks for *slug* with a fresh chunking of *content*.

        Idempotent: old chunks for this slug are always deleted first so stale
        fragments from a previous longer version of the page do not accumulate.
        """
        # Delete stale chunks before inserting new ones.
        self.delete_page(slug)

        chunks = chunk_markdown(content, self._max_chars, self._overlap)
        if not chunks:
            logger.debug("chunk_store_upsert_no_chunks", slug=slug)
            return

        chunk_texts = [ch_text for _, ch_text in chunks]
        try:
            vectors = self._llm.embed(chunk_texts, file_id=file_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("chunk_store_embed_failed", slug=slug, error=str(exc))
            return

        rows = [
            {
                "id": f"{slug}#{i:04d}",
                "slug": slug,
                "title": title,
                "section": section_name,
                "chunk_idx": i,
                "file_id": file_id,
                "document": ch_text,
                "embedding": vec,
            }
            for i, ((section_name, ch_text), vec) in enumerate(
                zip(chunks, vectors, strict=True)
            )
        ]

        try:
            with self._engine.begin() as conn:
                conn.execute(self._upsert_stmt(rows))
        except Exception as exc:  # noqa: BLE001
            logger.warning("chunk_store_upsert_failed", slug=slug, error=str(exc))
            return

        logger.debug("chunk_store_upserted", slug=slug, n_chunks=len(chunks))

    def delete_page(self, slug: str) -> None:
        """Remove all chunks for *slug*. No-op if none exist."""
        try:
            with self._engine.begin() as conn:
                conn.execute(delete(ChunkEmbedding).where(ChunkEmbedding.slug == slug))
            logger.debug("chunk_store_deleted", slug=slug)
        except Exception as exc:  # noqa: BLE001
            logger.warning("chunk_store_delete_failed", slug=slug, error=str(exc))

    def query(
        self,
        text: str,
        top_k: int = 10,
        file_id: str = "",
        file_ids: list[str] | None = None,
        usage_file_id: str = "",
    ) -> list[ChunkHit]:
        """Return the *top_k* most similar chunks to *text*.

        Args:
            text: Query string (e.g. a user question).
            top_k: Maximum number of chunks to return.
            file_id: When set (and *file_ids* is empty), restrict to this source file.
            file_ids: When set, restrict results to these source file IDs.
            usage_file_id: Correlation ID for embedding usage logging only — never
                applied as a metadata filter.

        Returns:
            List of ``ChunkHit`` sorted by descending similarity. Empty if the
            table is empty or the embeddings call fails.
        """
        embed_tag = usage_file_id or file_id or "chunk-query"
        if self.count() == 0:
            return []

        try:
            vectors = self._llm.embed([text], file_id=embed_tag)
        except Exception as exc:  # noqa: BLE001
            logger.warning("chunk_store_query_embed_failed", error=str(exc))
            return []

        qvec = vectors[0]
        distance = ChunkEmbedding.embedding.cosine_distance(qvec)
        stmt = select(
            ChunkEmbedding.slug,
            ChunkEmbedding.title,
            ChunkEmbedding.section,
            ChunkEmbedding.chunk_idx,
            ChunkEmbedding.document,
            ChunkEmbedding.file_id,
            distance.label("distance"),
        )
        if file_ids:
            stmt = stmt.where(ChunkEmbedding.file_id.in_(file_ids))
        elif file_id:
            stmt = stmt.where(ChunkEmbedding.file_id == file_id)
        stmt = stmt.order_by(distance).limit(top_k)

        try:
            with self._engine.connect() as conn:
                rows = conn.execute(stmt).all()
        except Exception as exc:  # noqa: BLE001
            logger.warning("chunk_store_query_failed", error=str(exc))
            return []

        hits = [
            ChunkHit(
                slug=r.slug,
                title=r.title or "",
                section=r.section or "",
                chunk_idx=int(r.chunk_idx),
                text=r.document,
                similarity=max(0.0, 1.0 - float(r.distance)),
                file_id=r.file_id or "",
            )
            for r in rows
        ]
        hits.sort(key=lambda h: h.similarity, reverse=True)
        return hits

    def count(self) -> int:
        """Return the total number of chunk embeddings."""
        with self._engine.connect() as conn:
            return int(
                conn.execute(select(func.count()).select_from(ChunkEmbedding)).scalar_one()
            )

    def count_by_file_id(self, file_id: str) -> int:
        """Return how many chunks are tagged with *file_id*."""
        with self._engine.connect() as conn:
            return int(
                conn.execute(
                    select(func.count())
                    .select_from(ChunkEmbedding)
                    .where(ChunkEmbedding.file_id == file_id)
                ).scalar_one()
            )

    def backfill_file_id(self, slug: str, file_id: str) -> int:
        """Set ``file_id`` on all existing chunks for *slug*.

        Returns:
            Number of chunk rows updated.
        """
        with self._engine.begin() as conn:
            result = conn.execute(
                update(ChunkEmbedding)
                .where(ChunkEmbedding.slug == slug)
                .values(file_id=file_id)
            )
        return int(result.rowcount or 0)

    def clear(self) -> None:
        """Delete ALL chunks. Used by ``scripts/reindex_chunks.py``."""
        with self._engine.begin() as conn:
            conn.execute(delete(ChunkEmbedding))
        self._write_meta()
        logger.info("chunk_store_cleared")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _upsert_stmt(self, rows: list[dict[str, object]]):  # type: ignore[no-untyped-def]
        """Build an INSERT ... ON CONFLICT(id) DO UPDATE statement."""
        ins = pg_insert(ChunkEmbedding).values(rows)
        return ins.on_conflict_do_update(
            index_elements=[ChunkEmbedding.id],
            set_={
                "slug": ins.excluded.slug,
                "title": ins.excluded.title,
                "section": ins.excluded.section,
                "chunk_idx": ins.excluded.chunk_idx,
                "file_id": ins.excluded.file_id,
                "document": ins.excluded.document,
                "embedding": ins.excluded.embedding,
            },
        )

    def _validate_meta(self) -> None:
        """Guard against a model/dim change without a reindex."""
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
                    f"chunk_embeddings were built with model {stored.model!r}, "
                    f"but config says {self._model!r}. "
                    "Run: docker compose exec api uv run python scripts/reindex_chunks.py"
                )
            if stored.dim != self._dim:
                raise EmbeddingModelMismatchError(
                    f"chunk_embeddings dimension {stored.dim} != config {self._dim}. "
                    "Run: docker compose exec api uv run python scripts/reindex_chunks.py"
                )

    def _write_meta(self) -> None:
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
