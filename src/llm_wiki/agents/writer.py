"""Writer Agent — creates or updates wiki pages from ingested documents.

Two scenarios, two separate prompts:
  A. create_page  — new topic, no existing page (LW-7)
  B. update_pages — enrich ≤5 existing pages (LW-8)
"""

from dataclasses import dataclass, field

from llm_wiki.agents.base import BaseAgent


@dataclass
class WikiPage:
    """A wiki page ready to be written to the filesystem."""

    slug: str
    title: str
    content: str
    source_files: list[str] = field(default_factory=list)


class WriterAgent(BaseAgent):
    """Generates and updates wiki pages using structured LLM output."""

    MAX_PAGES_PER_UPDATE = 5
    MAX_CONTENT_DROP_RATIO = 0.40

    async def run(self, *args: object, **kwargs: object) -> object:  # type: ignore[override]
        """Dispatch to create_page or update_pages depending on arguments."""
        raise NotImplementedError("Call create_page or update_pages directly")

    async def create_page(self, raw_content: str, file_id: str) -> WikiPage:
        """Create a brand-new wiki page from *raw_content*.

        Args:
            raw_content: Parsed plain text of the source document.
            file_id: UUID of the source file (for backlink tracking).

        Returns:
            A WikiPage with slug, title, and full markdown content.
        """
        raise NotImplementedError("Implemented in LW-7")

    async def update_pages(
        self,
        raw_content: str,
        existing_pages: list[WikiPage],
        file_id: str,
    ) -> list[WikiPage]:
        """Enrich up to 5 existing wiki pages with information from *raw_content*.

        Args:
            raw_content: Parsed plain text of the source document.
            existing_pages: Up to MAX_PAGES_PER_UPDATE pages to update.
            file_id: UUID of the source file.

        Returns:
            Updated WikiPage objects. Raises ValueError if >40% of any page's
            content would be removed.
        """
        if len(existing_pages) > self.MAX_PAGES_PER_UPDATE:
            raise ValueError(
                f"update_pages accepts at most {self.MAX_PAGES_PER_UPDATE} pages, "
                f"got {len(existing_pages)}"
            )
        raise NotImplementedError("Implemented in LW-8")
