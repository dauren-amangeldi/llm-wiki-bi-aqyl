"""Search Agent — finds relevant wiki pages for an incoming document.

v1 (LW-6): full index.md headings passed to LLM (no embedding pre-filter).
v2 (LW-12): embedding pre-filter (top 20) → LLM re-rank.
"""

import json
from dataclasses import dataclass

import structlog

from llm_wiki.agents.base import BaseAgent
from llm_wiki.config import settings
from llm_wiki.llm.client import LLMClient

logger = structlog.get_logger(__name__)


@dataclass
class SearchResult:
    """A single relevant wiki page with its relevance score."""

    slug: str
    title: str
    relevance_score: float


class SearchAgent(BaseAgent):
    """Finds existing wiki pages relevant to the content of an incoming file.

    v1 implementation sends the full index.md heading list to the LLM for
    relevance scoring.  Returns up to MAX_RESULTS sorted by descending score,
    or an empty list when all scores fall below the threshold (new topic).
    """

    RELEVANCE_THRESHOLD: float = 0.3
    MAX_RESULTS: int = 10
    # Approximate token budget for the document summary (~2 000 tokens × 4 chars)
    _SUMMARY_CHARS: int = 8_000

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialise the agent with an LLMClient instance.

        Args:
            llm_client: The shared LLM client used for all calls.
        """
        self._llm = llm_client

    async def run(  # type: ignore[override]
        self,
        file_text: str,
        index_headings: list[str],
        file_id: str = "",
    ) -> list[SearchResult]:
        """Find existing wiki pages relevant to *file_text*.

        Args:
            file_text: Parsed plain text of the uploaded document.
            index_headings: Heading strings extracted from index.md.
            file_id: Correlation ID used for usage tracking and structured logs.

        Returns:
            Ranked list of SearchResult objects (score ≥ threshold), at most
            MAX_RESULTS entries.  Empty list signals a brand-new topic.
        """
        if not index_headings:
            logger.debug("search_skipped_empty_index", file_id=file_id)
            return []

        summary = file_text[: self._SUMMARY_CHARS]
        headings_str = "\n".join(f"- {h}" for h in index_headings)

        prompt = self._llm.load_prompt(
            "search",
            language=settings.wiki_language,
            document_summary=summary,
            index_headings=headings_str,
        )
        system = "You are a wiki curator. Return only valid JSON."

        text, _usage = await self._llm.complete(
            prompt=prompt,
            system=system,
            file_id=file_id,
            agent_type="search",
            response_format="json",
        )

        candidates = _parse_search_response(text)

        results: list[SearchResult] = []
        for item in candidates:
            if not isinstance(item, dict):
                continue
            score = float(item.get("relevance_score", 0.0))
            if score < self.RELEVANCE_THRESHOLD:
                continue
            slug = str(item.get("slug", ""))
            title = str(item.get("title", slug))
            if not slug:
                continue
            results.append(SearchResult(slug=slug, title=title, relevance_score=score))

        results.sort(key=lambda r: r.relevance_score, reverse=True)
        final = results[: self.MAX_RESULTS]

        logger.info(
            "search_complete",
            file_id=file_id,
            candidates=len(candidates),
            returned=len(final),
        )
        return final


def _parse_search_response(raw: str) -> list[dict]:  # type: ignore[type-arg]
    """Extract the candidates array from an LLM response robustly.

    Tolerates the following formats that LLMs produce in practice:

    - ``{"candidates": [...]}``  — preferred (matches the prompt)
    - ``{"results": [...]}``     — common alias
    - ``[...]``                   — bare array (Ollama / older prompts)
    - ````` ```json\\n...\\n``` `````  — markdown fence wrapper

    Args:
        raw: Raw string returned by the LLM.

    Returns:
        List of raw candidate dicts.

    Raises:
        ValueError: If *raw* is not valid JSON or contains no extractable list.
    """
    text = raw.strip()

    # Strip markdown code fences (``` or ```json)
    if text.startswith("```"):
        parts = text.split("```")
        # parts[1] is the content block
        inner = parts[1] if len(parts) > 1 else text
        if inner.startswith("json"):
            inner = inner[4:]
        text = inner.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Search Agent: invalid JSON from LLM: {exc}") from exc

    # Bare array — accepted for backward compatibility (e.g. Ollama)
    if isinstance(data, list):
        return data  # type: ignore[return-value]

    if isinstance(data, dict):
        # Check known wrapper keys in priority order
        for key in ("candidates", "results", "items", "matches", "pages"):
            value = data.get(key)
            if isinstance(value, list):
                return value  # type: ignore[return-value]

        # Single candidate returned as a flat object {"slug": ..., ...}
        if "slug" in data:
            return [data]  # type: ignore[return-value]

    raise ValueError(
        f"Search Agent: cannot extract candidates array from LLM response: {raw[:300]!r}"
    )
