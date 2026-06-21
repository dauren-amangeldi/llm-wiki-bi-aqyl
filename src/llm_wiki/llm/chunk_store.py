"""ChromaDB-backed chunk store for wiki page *bodies* (LW-20.1).

Why a second collection instead of reusing ``headings``?
---------------------------------------------------------
``EmbeddingStore`` (``headings`` collection) stores one embedding per page
**title**. It was designed for the *SearchAgent*: "given a new document,
which existing wiki pages should be updated?"  That is a page-level
classification task that works perfectly with title embeddings.

``ChunkStore`` (``chunks`` collection) stores one embedding per ~500-token
**text fragment** of a page body.  It is designed for the *AnswerAgent*:
"given a user question, which paragraph actually contains the answer?"
That is a passage-level retrieval task where title embeddings have poor
recall — a page titled "Training Runs" may contain the only mention of
"Adam optimizer" in the whole wiki, but its title embedding will never be
the nearest neighbour for the query "what optimizer do we use?".

The two tasks require different granularity, different top-k thresholds,
and different ID namespaces.  Merging them into one collection would
degrade both: heading-vs-chunk similarity scores are not comparable, and
the SearchAgent would start returning chunk IDs (``slug#0003``) instead
of page slugs.

Collection name: ``chunks`` (never ``headings``).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
import structlog

from llm_wiki.llm.embeddings import EmbeddingModelMismatchError

logger = structlog.get_logger(__name__)

_COLLECTION_NAME = "chunks"
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

    Uses the heading regex directly rather than ``parsers/markdown.py``'s
    token stream because we need the *text span* of each section, not just
    the heading metadata.

    Args:
        text: Raw Markdown page content.
        max_chars: Maximum characters per chunk.
        overlap: Character overlap between consecutive chunks from the same
            long section.

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
    """ChromaDB-backed chunk index for wiki page bodies.

    Stores one embedding per text chunk (≈500 tokens) so that the
    ``AnswerAgent`` can retrieve the specific paragraphs that contain the
    answer to a user question, rather than just the closest page title.

    Design principles mirror ``EmbeddingStore``:
    - Synchronous: ChromaDB's Python client is synchronous.
    - Idempotent: ``upsert_page`` deletes old chunks before inserting new ones.
    - Isolated: failures are logged and NOT propagated (wiki/*.md is the
      source of truth; Chroma is a cache).
    """

    def __init__(
        self,
        chroma_path: Path,
        llm_client: "LLMClient",  # noqa: F821
        chroma_client: chromadb.ClientAPI | None = None,
    ) -> None:
        """Initialise and validate the ChromaDB chunks collection.

        Args:
            chroma_path: Directory for ChromaDB persistence.
            llm_client: Used to call the embeddings API.
            chroma_client: Optional pre-constructed client (injected in tests).

        Raises:
            EmbeddingModelMismatchError: If the stored collection was built
                with a different embedding model or dimension.
        """
        from llm_wiki.config import settings
        from llm_wiki.llm.client import LLMClient  # noqa: PLC0415

        self._llm: LLMClient = llm_client  # type: ignore[assignment]
        self._model: str = settings.embedding_model
        self._dim: int = settings.embedding_dimensions
        self._max_chars: int = settings.chunk_max_chars
        self._overlap: int = settings.chunk_overlap_chars

        if chroma_client is not None:
            self._chroma = chroma_client
        else:
            chroma_path.mkdir(parents=True, exist_ok=True)
            self._chroma = chromadb.PersistentClient(path=str(chroma_path))

        self._col = self._get_or_create_collection()

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

        Idempotent: calling with the same content produces the same result.
        Old chunks for this slug are always deleted first so stale fragments
        from a previous longer version of the page do not accumulate.

        Args:
            slug: Page identifier (e.g. ``"transformers"``).
            title: Human-readable page title (stored in metadata for UI).
            content: Full markdown content of the wiki page.
            file_id: Correlation ID for usage tracking.
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
            logger.warning(
                "chunk_store_embed_failed", slug=slug, error=str(exc)
            )
            return

        ids = [f"{slug}#{i:04d}" for i in range(len(chunks))]
        metadatas = [
            {
                "slug": slug,
                "title": title,
                "section": section_name,
                "chunk_idx": i,
                "file_id": file_id,
            }
            for i, (section_name, _) in enumerate(chunks)
        ]

        try:
            self._col.upsert(
                ids=ids,
                embeddings=vectors,
                metadatas=metadatas,
                documents=chunk_texts,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "chunk_store_chroma_upsert_failed", slug=slug, error=str(exc)
            )
            return

        logger.debug("chunk_store_upserted", slug=slug, n_chunks=len(chunks))

    def delete_page(self, slug: str) -> None:
        """Remove all chunks for *slug*.  No-op if none exist.

        Args:
            slug: Page identifier whose chunks should be removed.
        """
        try:
            self._col.delete(where={"slug": slug})
            logger.debug("chunk_store_deleted", slug=slug)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "chunk_store_delete_failed", slug=slug, error=str(exc)
            )

    def query(
        self,
        text: str,
        top_k: int = 10,
        file_id: str = "",
        file_ids: list[str] | None = None,
    ) -> list[ChunkHit]:
        """Return the *top_k* most similar chunks to *text*.

        Args:
            text: Query string (e.g. a user question).
            top_k: Maximum number of chunks to return.
            file_id: Correlation ID for embedding usage tracking and optional filter.
            file_ids: When set, restrict results to these source file IDs.

        Returns:
            List of ``ChunkHit`` sorted by descending similarity.  Empty if
            the collection is empty or the embeddings call fails.
        """
        where_filter: dict[str, object] | None = None
        if file_ids:
            where_filter = {"file_id": {"$in": file_ids}}
        elif file_id:
            where_filter = {"file_id": file_id}

        current_count = self.count()
        if current_count == 0:
            return []

        n = min(top_k, current_count)
        try:
            vectors = self._llm.embed([text], file_id=file_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("chunk_store_query_embed_failed", error=str(exc))
            return []

        try:
            results = self._col.query(
                query_embeddings=[vectors[0]],
                n_results=n,
                where=where_filter,  # type: ignore[arg-type]
                include=["metadatas", "distances", "documents"],  # type: ignore[list-item]
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("chunk_store_query_failed", error=str(exc))
            return []

        hits: list[ChunkHit] = []
        ids: list[str] = results.get("ids", [[]])[0]  # type: ignore[index]
        distances: list[float] = results.get("distances", [[]])[0]  # type: ignore[index]
        metas: list[dict[str, object]] = results.get("metadatas", [[]])[0]  # type: ignore[index]
        docs: list[str] = results.get("documents", [[]])[0]  # type: ignore[index]

        for _id, dist, meta, doc in zip(ids, distances, metas, docs, strict=True):
            similarity = max(0.0, 1.0 - float(dist))
            hits.append(
                ChunkHit(
                    slug=str(meta.get("slug", "")),
                    title=str(meta.get("title", "")),
                    section=str(meta.get("section", "")),
                    chunk_idx=int(meta.get("chunk_idx", 0)),
                    text=doc,
                    similarity=similarity,
                    file_id=str(meta.get("file_id", "")),
                )
            )

        hits.sort(key=lambda h: h.similarity, reverse=True)
        return hits

    def count(self) -> int:
        """Return the total number of chunk embeddings in the collection."""
        return self._col.count()  # type: ignore[no-any-return]

    def count_by_file_id(self, file_id: str) -> int:
        """Return how many chunks are tagged with *file_id*."""
        result = self._col.get(where={"file_id": file_id}, include=[])
        return len(result.get("ids", []))

    def backfill_file_id(self, slug: str, file_id: str) -> int:
        """Set ``file_id`` metadata on all existing chunks for *slug*.

        Returns:
            Number of chunk records updated.
        """
        existing = self._col.get(where={"slug": slug}, include=["metadatas"])
        ids: list[str] = existing.get("ids", [])
        if not ids:
            return 0
        metas: list[dict[str, object]] = existing.get("metadatas", [])
        updated_metas: list[dict[str, object]] = []
        for meta in metas:
            merged = dict(meta)
            merged["file_id"] = file_id
            updated_metas.append(merged)
        self._col.update(ids=ids, metadatas=updated_metas)
        return len(ids)

    def clear(self) -> None:
        """Delete ALL chunks.  Used by ``scripts/reindex_chunks.py``."""
        self._chroma.delete_collection(_COLLECTION_NAME)
        self._col = self._get_or_create_collection(fresh=True)
        logger.info("chunk_store_cleared")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_or_create_collection(
        self, fresh: bool = False
    ) -> chromadb.Collection:
        """Return (or create) the chunks collection, validating model metadata."""
        wanted_meta = {
            "hnsw:space": "cosine",
            "embedding_model": self._model,
            "embedding_dim": str(self._dim),
        }

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
                    f"chunks collection was built with model {stored_model!r}, "
                    f"but config says {self._model!r}. "
                    "Run: docker compose exec api uv run python scripts/reindex_chunks.py"
                )
            if stored_dim and stored_dim != str(self._dim):
                raise EmbeddingModelMismatchError(
                    f"chunks collection dimension {stored_dim} != config {self._dim}. "
                    "Run: docker compose exec api uv run python scripts/reindex_chunks.py"
                )

        return col
