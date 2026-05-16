"""Search Agent — finds relevant wiki pages for an incoming document.

v1 (LW-6): full index.md passed to LLM (no embedding pre-filter).
v2 (LW-12): embedding pre-filter (top 20) → LLM re-rank.
"""

from dataclasses import dataclass

from llm_wiki.agents.base import BaseAgent


@dataclass
class SearchResult:
    """A single relevant wiki page with its relevance score."""

    slug: str
    title: str
    relevance_score: float


class SearchAgent(BaseAgent):
    """Finds existing wiki pages relevant to the content of an incoming file.

    Returns 3–10 results, or an empty list when the document describes a
    brand-new topic (all scores below threshold 0.3).
    """

    RELEVANCE_THRESHOLD = 0.3
    MAX_RESULTS = 10

    async def run(  # type: ignore[override]
        self,
        file_text: str,
        index_headings: list[str],
    ) -> list[SearchResult]:
        """Find relevant wiki pages for *file_text*.

        Args:
            file_text: Parsed plain text of the uploaded document.
            index_headings: List of heading strings extracted from index.md.

        Returns:
            Ranked list of relevant wiki pages, empty if none exceed the
            relevance threshold.
        """
        raise NotImplementedError("Implemented in LW-6")
