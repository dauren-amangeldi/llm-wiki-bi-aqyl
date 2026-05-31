"""Parser and writer for index.md — the wiki knowledge map.

Uses filelock (cross-platform) to prevent concurrent writes from corrupting the file.
All mutating operations follow: lock → read → modify → atomic_write → release.

ChromaDB synchronisation (optional):
    Pass an ``EmbeddingStore`` to ``IndexStorage.__init__`` to keep the vector
    index in sync.  Failures in ChromaDB are logged and swallowed — index.md is
    the authoritative source of truth; the Chroma collection is a derived cache.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from filelock import FileLock

from llm_wiki.storage.filesystem import atomic_write

if TYPE_CHECKING:
    from llm_wiki.llm.embeddings import EmbeddingStore

logger = structlog.get_logger(__name__)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_WIKI_LINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
_LIST_ENTRY_RE = re.compile(r"^\s*-\s*\[\[([^\]]+)\]\]")


@dataclass
class Heading:
    """A heading entry in index.md."""

    level: int
    text: str
    slug: str | None = None


class IndexStorage:
    """Read and write wiki headings in index.md.

    All mutating operations acquire an exclusive file lock before reading
    the current state, applying the change, and writing back atomically.

    An optional ``EmbeddingStore`` keeps ChromaDB in sync with every
    add/move/remove operation.  If the store is absent (or raises), index.md
    is still updated correctly.
    """

    def __init__(
        self,
        index_path: Path,
        embedding_store: EmbeddingStore | None = None,
    ) -> None:
        """Initialise with the path to index.md.

        Args:
            index_path: Absolute path to the index.md file.
            embedding_store: Optional ChromaDB store for vector sync.  When
                provided, every add/move/remove mirrors to Chroma.  Failures
                in Chroma are logged and do not propagate.
        """
        self._path = index_path
        self._embedding_store: EmbeddingStore | None = embedding_store
        # Lock file lives alongside index.md; FileLock is cross-platform
        self._lock = FileLock(str(index_path) + ".lock", timeout=30)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read_headings(self) -> list[Heading]:
        """Return all headings currently in index.md.

        Returns:
            Ordered list of Heading objects parsed from the file.
            Empty list if the file does not exist.
        """
        if not self._path.exists():
            return []
        headings: list[Heading] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            m = _HEADING_RE.match(line)
            if m:
                level = len(m.group(1))
                text = m.group(2).strip()
                links = _WIKI_LINK_RE.findall(text)
                slug = links[0] if links else None
                headings.append(Heading(level=level, text=text, slug=slug))
        return headings

    def add_page(self, slug: str, section: str, title: str = "", level: int = 2) -> None:
        """Append a new page entry under *section* in index.md.

        Creates the section if it doesn't exist.  Idempotent: calling with
        the same *slug* twice does not add a duplicate entry.  On success,
        upserts the heading into ChromaDB (if an ``EmbeddingStore`` is set).

        Args:
            slug: Page slug written as ``[[slug]]`` in the list entry.
            section: Section heading text (without ``##`` prefix).
            title: Human-readable title for the embedding.  Defaults to slug.
            level: Markdown heading level for the Chroma metadata.
        """
        with self._lock:
            content = self._read()
            content = self._add_entry(content, slug, section)
            self._write(content)

        # ChromaDB sync — happens AFTER the lock is released and index.md is
        # safely written, so a Chroma failure can never corrupt the index.
        effective_title = title or slug
        if self._embedding_store is not None:
            try:
                self._embedding_store.upsert_heading(
                    slug=slug,
                    title=effective_title,
                    section=section,
                    level=level,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "embedding_sync_failed_on_add",
                    slug=slug,
                    error=str(exc),
                )

    def move_page(self, slug: str, new_section: str) -> None:
        """Move an existing page to a different section.

        Removes the entry from its current location and re-inserts it under
        *new_section*.  If the slug does not exist in the file the call is a
        no-op.  Updates only the *section* metadata in ChromaDB (embedding
        vector is not recomputed — the title has not changed).

        Args:
            slug: Page slug to relocate.
            new_section: Target section heading text (without ``##`` prefix).
        """
        with self._lock:
            content = self._read()
            content = self._remove_entry(content, slug)
            content = self._add_entry(content, slug, new_section)
            self._write(content)

        if self._embedding_store is not None:
            try:
                self._embedding_store.update_metadata(slug=slug, section=new_section)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "embedding_sync_failed_on_move",
                    slug=slug,
                    error=str(exc),
                )

    def remove_page(self, slug: str) -> None:
        """Remove an existing page entry from index.md and ChromaDB.

        No-op if *slug* is not present.

        Args:
            slug: Page slug to remove.
        """
        with self._lock:
            content = self._read()
            content = self._remove_entry(content, slug)
            self._write(content)

        if self._embedding_store is not None:
            try:
                self._embedding_store.delete(slug=slug)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "embedding_sync_failed_on_remove",
                    slug=slug,
                    error=str(exc),
                )

    def get_backlinks(self, slug: str) -> list[str]:
        """Return slugs of other pages that appear on the same line as *slug*.

        Scans index.md for every line that contains ``[[slug]]`` and collects
        any other ``[[other]]`` references found on that same line.  This
        captures inline cross-references written into section entries, e.g.
        ``- [[page-a]] — see also [[slug]]``.

        Args:
            slug: The target page slug.

        Returns:
            Deduplicated list of co-occurring slugs, preserving first-seen order.
        """
        if not self._path.exists():
            return []
        target = f"[[{slug}]]"
        content = self._path.read_text(encoding="utf-8")
        backlinks: list[str] = []
        for line in content.splitlines():
            if target not in line:
                continue
            for found in _WIKI_LINK_RE.findall(line):
                if found != slug and found not in backlinks:
                    backlinks.append(found)
        return backlinks

    # ------------------------------------------------------------------
    # Private helpers (no lock — always called while already holding one)
    # ------------------------------------------------------------------

    def _read(self) -> str:
        """Return raw file content, or empty string if file absent."""
        return self._path.read_text(encoding="utf-8") if self._path.exists() else ""

    def _write(self, content: str) -> None:
        """Atomically persist *content* to index.md."""
        atomic_write(self._path, content)

    def _add_entry(self, content: str, slug: str, section: str) -> str:
        """Return new content with ``[[slug]]`` added under *section*.

        Idempotent: if ``[[slug]]`` already appears anywhere in the content
        the function returns *content* unchanged.

        Args:
            content: Current raw file content.
            slug: Slug to add.
            section: Section heading text (no ``##`` prefix).

        Returns:
            Updated content string.
        """
        if f"[[{slug}]]" in content:
            return content  # already present

        entry = f"- [[{slug}]]\n"
        section_header = f"## {section}"
        lines = content.splitlines(keepends=True)

        # Try to find the section heading
        section_idx: int | None = None
        for i, line in enumerate(lines):
            if line.rstrip("\n") == section_header:
                section_idx = i
                break

        if section_idx is None:
            # Section doesn't exist — append at the end
            if content and not content.endswith("\n"):
                content += "\n"
            return content + f"\n{section_header}\n{entry}"

        # Find end of this section: the next line starting with #
        insert_at = len(lines)
        for i in range(section_idx + 1, len(lines)):
            if lines[i].lstrip().startswith("#"):
                insert_at = i
                break

        lines.insert(insert_at, entry)
        return "".join(lines)

    def _remove_entry(self, content: str, slug: str) -> str:
        """Return new content with the ``- [[slug]]`` list entry removed.

        Only removes exact list entries (``- [[slug]]`` pattern).  Inline
        references on other lines are not touched.

        Args:
            content: Current raw file content.
            slug: Slug whose entry should be removed.

        Returns:
            Updated content string.
        """
        pattern = re.compile(
            rf"^\s*-\s*\[\[{re.escape(slug)}\]\][^\n]*\n?",
            re.MULTILINE,
        )
        return pattern.sub("", content)
