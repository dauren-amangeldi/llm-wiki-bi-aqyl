"""Writer Agent — creates or updates wiki pages from ingested documents.

Two scenarios, two separate prompts:
  A. create_page  — new topic, no existing page (LW-7)
  B. update_pages — enrich ≤5 existing pages (LW-8)
"""

import json
from dataclasses import dataclass, field

import structlog

from llm_wiki.agents.base import BaseAgent
from llm_wiki.llm.client import LLMClient
from llm_wiki.utils.slugify import is_valid_slug, to_slug

logger = structlog.get_logger(__name__)


@dataclass
class WikiPage:
    """A wiki page ready to be written to the filesystem."""

    slug: str
    title: str
    content: str
    source_files: list[str] = field(default_factory=list)


class WriterAgent(BaseAgent):
    """Generates and updates wiki pages using structured LLM output.

    Uses separate prompts for creation (writer_create.md) and updates
    (writer_update.md).  All output is JSON-structured to avoid free-form
    markdown hallucination.
    """

    MAX_PAGES_PER_UPDATE: int = 5
    MAX_CONTENT_DROP_RATIO: float = 0.40

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialise the agent with an LLMClient instance.

        Args:
            llm_client: The shared LLM client used for all calls.
        """
        self._llm = llm_client

    async def run(self, *args: object, **kwargs: object) -> object:  # type: ignore[override]
        """Not used directly — call create_page or update_pages instead."""
        raise NotImplementedError("Call create_page or update_pages directly")

    # ------------------------------------------------------------------
    # Scenario A — create a brand-new wiki page
    # ------------------------------------------------------------------

    async def create_page(self, raw_content: str, file_id: str) -> WikiPage:
        """Create a brand-new wiki page from *raw_content*.

        Sends the full source text to the LLM with the writer_create prompt
        and expects a JSON response with ``slug``, ``title``, and ``content``
        fields.  The slug is validated and re-derived from the title if the
        LLM returns an invalid one.

        Args:
            raw_content: Parsed plain text of the source document.
            file_id: UUID of the source file (used for usage tracking).

        Returns:
            A WikiPage with slug, title, and full markdown content.

        Raises:
            ValueError: If the LLM response is invalid JSON, missing content,
                or if a valid slug cannot be produced.
        """
        prompt = self._llm.load_prompt("writer_create", raw_content=raw_content)
        system = "You are a technical wiki author. Return only valid JSON."

        text, _usage = await self._llm.complete(
            prompt=prompt,
            system=system,
            file_id=file_id,
            agent_type="writer",
            response_format="json",
        )

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Writer Agent: invalid JSON from LLM: {exc}") from exc

        slug = str(data.get("slug", "")).strip()
        title = str(data.get("title", "")).strip()
        content = str(data.get("content", "")).strip()

        if not content:
            raise ValueError("Writer Agent: LLM returned empty content")

        # Ensure slug is valid kebab-case; derive from title if not
        if not is_valid_slug(slug):
            slug = to_slug(title)
        if not is_valid_slug(slug):
            raise ValueError(
                f"Writer Agent: cannot produce a valid slug from title={title!r}"
            )

        logger.info("page_created", file_id=file_id, slug=slug)
        return WikiPage(slug=slug, title=title, content=content, source_files=[file_id])

    # ------------------------------------------------------------------
    # Scenario B — update up to 5 existing wiki pages
    # ------------------------------------------------------------------

    async def update_pages(
        self,
        raw_content: str,
        existing_pages: list[WikiPage],
        file_id: str,
    ) -> list[WikiPage]:
        """Enrich up to 5 existing wiki pages with information from *raw_content*.

        Calls the LLM once per page using the writer_update prompt.  Rejects
        any update that would remove more than MAX_CONTENT_DROP_RATIO of the
        original content.

        Args:
            raw_content: Parsed plain text of the source document.
            existing_pages: Pages to update; at most MAX_PAGES_PER_UPDATE.
            file_id: UUID of the source file.

        Returns:
            List of updated WikiPage objects in the same order as input.

        Raises:
            ValueError: If more than MAX_PAGES_PER_UPDATE pages are supplied,
                if the LLM returns invalid JSON, or if an update would drop
                too much content.
        """
        if len(existing_pages) > self.MAX_PAGES_PER_UPDATE:
            raise ValueError(
                f"update_pages accepts at most {self.MAX_PAGES_PER_UPDATE} pages, "
                f"got {len(existing_pages)}"
            )

        updated: list[WikiPage] = []

        for page in existing_pages:
            prompt = self._llm.load_prompt(
                "writer_update",
                slug=page.slug,
                existing_content=page.content,
                raw_content=raw_content,
            )
            system = "You are a technical wiki editor. Return only valid JSON."

            text, _usage = await self._llm.complete(
                prompt=prompt,
                system=system,
                file_id=file_id,
                agent_type="writer",
                response_format="json",
            )

            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Writer Agent: invalid JSON for page '{page.slug}': {exc}"
                ) from exc

            new_content = str(data.get("content", "")).strip()
            new_title = str(data.get("title", page.title)).strip() or page.title

            if self._check_content_drop(page.content, new_content):
                raise ValueError(
                    f"Writer Agent: update to '{page.slug}' would remove "
                    f">{self.MAX_CONTENT_DROP_RATIO:.0%} of content — rejected"
                )

            merged_sources = list(dict.fromkeys(page.source_files + [file_id]))
            updated.append(
                WikiPage(
                    slug=page.slug,
                    title=new_title,
                    content=new_content,
                    source_files=merged_sources,
                )
            )
            logger.info("page_updated", file_id=file_id, slug=page.slug)

        return updated

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_content_drop(self, old_content: str, new_content: str) -> bool:
        """Return True if *new_content* removes more than the allowed ratio.

        Args:
            old_content: Original page text.
            new_content: Proposed replacement text.

        Returns:
            True when the drop ratio exceeds MAX_CONTENT_DROP_RATIO.
        """
        old_len = len(old_content.strip())
        if old_len == 0:
            return False
        new_len = len(new_content.strip())
        drop_ratio = 1.0 - (new_len / old_len)
        return drop_ratio > self.MAX_CONTENT_DROP_RATIO
