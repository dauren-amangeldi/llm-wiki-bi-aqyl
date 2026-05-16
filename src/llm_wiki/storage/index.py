"""Parser and writer for index.md — the wiki knowledge map.

Uses fcntl file locking to prevent concurrent writes from corrupting the file.
Implemented in LW-2.
"""

from dataclasses import dataclass
from pathlib import Path


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
    """

    def __init__(self, index_path: Path) -> None:
        """Initialise with the path to index.md.

        Args:
            index_path: Absolute path to the index.md file.
        """
        self._path = index_path

    def read_headings(self) -> list[Heading]:
        """Return all headings currently in index.md.

        Returns:
            Ordered list of Heading objects parsed from the file.
        """
        raise NotImplementedError("Implemented in LW-2")

    def add_page(self, slug: str, section: str) -> None:
        """Append a new page entry under *section* in index.md.

        Args:
            slug: Page slug (becomes the link target).
            section: Section heading under which the page is listed.
        """
        raise NotImplementedError("Implemented in LW-2")

    def move_page(self, slug: str, new_section: str) -> None:
        """Move an existing page to a different section.

        Args:
            slug: Page slug to relocate.
            new_section: Target section heading.
        """
        raise NotImplementedError("Implemented in LW-2")

    def get_backlinks(self, slug: str) -> list[str]:
        """Return slugs of all pages that link to *slug*.

        Args:
            slug: The target page slug.

        Returns:
            List of slugs with an incoming [[slug]] link.
        """
        raise NotImplementedError("Implemented in LW-2")
