"""Wiki knowledge-map index — backed by the ``wiki_index`` table in Postgres.

IndexStorage keeps the list of wiki pages (slug → section) and mirrors every
add/move/remove to the pgvector heading store. It used to persist ``index.md``
on local disk; the index now lives in Postgres so the app pods stay stateless.

The public interface is unchanged so existing callers (pipeline, linter,
reindex) keep working. The ``index_path`` constructor argument is accepted for
backwards compatibility but ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from llm_wiki.storage.metadata import WikiIndexEntry, get_sync_engine

if TYPE_CHECKING:
    from llm_wiki.llm.embeddings import EmbeddingStore

logger = structlog.get_logger(__name__)


@dataclass
class Heading:
    """A heading entry in the wiki index."""

    level: int
    text: str
    slug: str | None = None


class IndexStorage:
    """Read and write wiki page entries in the ``wiki_index`` table.

    An optional ``EmbeddingStore`` keeps pgvector in sync with every
    add/move/remove operation. Failures in pgvector are logged and swallowed —
    the index table is the authoritative source; the vectors are a cache.
    """

    def __init__(
        self,
        index_path: Path | None = None,
        embedding_store: EmbeddingStore | None = None,
    ) -> None:
        """Initialise the index store.

        Args:
            index_path: Ignored (kept for backwards-compatible call sites).
            embedding_store: Optional pgvector store mirrored on every mutation.
        """
        self._embedding_store: EmbeddingStore | None = embedding_store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read_headings(self) -> list[Heading]:
        """Return the index as headings: a section header then its page entries."""
        headings: list[Heading] = []
        seen_sections: set[str] = set()
        for slug, _title, level, section in self.read_pages():
            if section not in seen_sections:
                seen_sections.add(section)
                headings.append(Heading(level=2, text=section, slug=None))
            headings.append(Heading(level=level, text=f"[[{slug}]]", slug=slug))
        return headings

    def read_pages(self) -> list[tuple[str, str, int, str]]:
        """Return ``(slug, title, level, section)`` for every indexed page."""
        with get_sync_engine().connect() as conn:
            rows = conn.execute(
                select(
                    WikiIndexEntry.slug,
                    WikiIndexEntry.title,
                    WikiIndexEntry.level,
                    WikiIndexEntry.section,
                ).order_by(WikiIndexEntry.section, WikiIndexEntry.slug)
            ).all()
        return [(str(r[0]), str(r[1]), int(r[2]), str(r[3])) for r in rows]

    def add_page(self, slug: str, section: str, title: str = "", level: int = 2) -> None:
        """Add or update a page entry under *section*; mirror to pgvector."""
        effective_title = title or slug
        with get_sync_engine().begin() as conn:
            ins = pg_insert(WikiIndexEntry).values(
                slug=slug, title=effective_title, section=section, level=level
            )
            conn.execute(
                ins.on_conflict_do_update(
                    index_elements=[WikiIndexEntry.slug],
                    set_={
                        "title": ins.excluded.title,
                        "section": ins.excluded.section,
                        "level": ins.excluded.level,
                    },
                )
            )

        if self._embedding_store is not None:
            try:
                self._embedding_store.upsert_heading(
                    slug=slug, title=effective_title, section=section, level=level
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("embedding_sync_failed_on_add", slug=slug, error=str(exc))

    def move_page(self, slug: str, new_section: str) -> None:
        """Move a page to a different section; mirror to pgvector."""
        with get_sync_engine().begin() as conn:
            conn.execute(
                update(WikiIndexEntry)
                .where(WikiIndexEntry.slug == slug)
                .values(section=new_section)
            )

        if self._embedding_store is not None:
            try:
                self._embedding_store.update_metadata(slug=slug, section=new_section)
            except Exception as exc:  # noqa: BLE001
                logger.warning("embedding_sync_failed_on_move", slug=slug, error=str(exc))

    def remove_page(self, slug: str) -> None:
        """Remove a page entry; mirror to pgvector. No-op if absent."""
        with get_sync_engine().begin() as conn:
            conn.execute(delete(WikiIndexEntry).where(WikiIndexEntry.slug == slug))

        if self._embedding_store is not None:
            try:
                self._embedding_store.delete(slug=slug)
            except Exception as exc:  # noqa: BLE001
                logger.warning("embedding_sync_failed_on_remove", slug=slug, error=str(exc))

    def get_backlinks(self, slug: str) -> list[str]:
        """Kept for interface compatibility. Page backlinks come from the wiki
        store / backlinks_sync, so index-level co-occurrence is not tracked."""
        return []
