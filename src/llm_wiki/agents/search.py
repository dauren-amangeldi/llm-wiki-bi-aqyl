"""Search Agent v2 — embedding pre-filter + LLM re-rank.

Two-stage pipeline:
    1. Embedding pre-filter: cosine similarity of the document summary against
       all indexed headings in ChromaDB → top SEARCH_TOP_K candidates.
    2. LLM re-rank: GPT-5.4 Mini reads the summary and ≤20 candidates and
       returns 0–SEARCH_FINAL_K_MAX pages with rerank scores and reasoning.

v1 historic prompt preserved in docs/prompts.md (LW-6).
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import structlog

from llm_wiki.agents.base import BaseAgent
from llm_wiki.config import settings
from llm_wiki.llm.client import LLMClient
from llm_wiki.llm.embeddings import EmbeddingStore, SearchHit
from llm_wiki.utils.summary import extract_summary

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)


class SearchAgentError(RuntimeError):
    """Raised when the Search Agent cannot recover from an error."""


class SearchAgent(BaseAgent):
    """Find existing wiki pages relevant to a newly uploaded document.

    Uses a two-stage approach:
    1. Vector similarity pre-filter (ChromaDB) to narrow the candidate pool.
    2. LLM re-rank to score and justify the final selection.

    Returns an empty list when no pages meet the relevance threshold — this
    is a valid signal that the document introduces a brand-new topic.
    On LLM timeout/error, falls back to the top-N embedding-only results with
    ``rerank_score=None`` rather than failing the entire pipeline.
    """

    def __init__(self, llm_client: LLMClient, embedding_store: EmbeddingStore) -> None:
        """Initialise with an LLM client and an EmbeddingStore.

        Args:
            llm_client: Used for the LLM re-rank step.
            embedding_store: Used for the embedding pre-filter step.
        """
        self._llm = llm_client
        self._store = embedding_store

    async def run(  # type: ignore[override]
        self,
        file_text: str,
        index_headings: list[str],  # kept for API compat; ignored in v2
        file_id: str = "",
    ) -> list[SearchHit]:
        """Thin shim — delegates to ``search()``.

        The *index_headings* parameter is accepted but ignored in v2; candidates
        are fetched directly from ChromaDB.  Kept so the orchestrator call-site
        (``search_agent.run(file_text, heading_texts, file_id=…)``) does not
        need to change.

        Args:
            file_text: Full parsed text of the uploaded document.
            index_headings: Ignored in v2 (ChromaDB is the source).
            file_id: Correlation ID for usage tracking and structured logs.

        Returns:
            Ranked list of SearchHit objects.
        """
        return await self.search(file_text, file_id=file_id)

    async def search(
        self,
        file_text: str,
        file_id: str = "",
    ) -> list[SearchHit]:
        """Find relevant wiki pages for *file_text*.

        Args:
            file_text: Full parsed text of the uploaded document.
            file_id: Correlation ID for usage tracking and structured logs.

        Returns:
            Ranked list of SearchHit objects (at most SEARCH_FINAL_K_MAX).
            Empty list means no relevant pages were found — the document is a
            brand-new topic.
        """
        summary = extract_summary(file_text, max_chars=settings.search_summary_max_chars)

        # ------------------------------------------------------------------
        # Stage 1: embedding pre-filter
        # ------------------------------------------------------------------
        if self._store.count() == 0:
            logger.debug("search_skipped_empty_store", file_id=file_id)
            return []

        try:
            logger.info(
                "search_embedding_prefilter_start",
                file_id=file_id,
                store_count=self._store.count(),
            )
            candidates = self._store.query(
                summary, top_k=settings.search_top_k, file_id=file_id
            )
            logger.info(
                "search_embedding_prefilter_done",
                file_id=file_id,
                n_candidates=len(candidates),
            )
        except Exception as exc:
            raise SearchAgentError(f"Embedding query failed: {exc}") from exc

        above_threshold = [
            c for c in candidates if c.similarity >= settings.search_similarity_threshold
        ]

        if not above_threshold:
            logger.debug(
                "search_all_below_threshold",
                file_id=file_id,
                n_candidates=len(candidates),
                threshold=settings.search_similarity_threshold,
            )
            return []

        # ------------------------------------------------------------------
        # Stage 2: LLM re-rank
        # ------------------------------------------------------------------
        try:
            logger.info(
                "search_llm_rerank_start",
                file_id=file_id,
                n_candidates=len(above_threshold),
            )
            reranked = await self._llm_rerank(
                summary=summary,
                candidates=above_threshold,
                file_id=file_id,
            )
        except Exception as exc:
            # Graceful degradation: return top-5 embedding results without rerank
            logger.warning(
                "search_llm_rerank_failed_fallback",
                file_id=file_id,
                error=str(exc),
                fallback_count=min(5, len(above_threshold)),
            )
            return above_threshold[: min(5, len(above_threshold))]

        logger.info(
            "search_complete",
            file_id=file_id,
            n_embedding_candidates=len(above_threshold),
            n_final=len(reranked),
        )
        return reranked

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _llm_rerank(
        self,
        summary: str,
        candidates: list[SearchHit],
        file_id: str,
    ) -> list[SearchHit]:
        """Call the LLM to re-rank *candidates* against *summary*.

        Args:
            summary: Compact document summary.
            candidates: Pre-filtered SearchHit objects from embedding stage.
            file_id: Correlation ID.

        Returns:
            Re-ranked SearchHit list with rerank_score and reason populated.
            May be empty if the LLM decides nothing is relevant.
        """
        candidates_text = "\n".join(
            f"{i + 1}. slug={c.slug!r}  title={c.title!r}  "
            f"section={c.section!r}  similarity={c.similarity:.2f}"
            for i, c in enumerate(candidates)
        )

        prompt = self._llm.load_prompt(
            "search",
            language=settings.wiki_language,
            document_summary=summary,
            candidates=candidates_text,
        )

        text, _usage = await self._llm.complete(
            prompt=prompt,
            system="You are a wiki curator. Return only valid JSON.",
            file_id=file_id,
            agent_type="search",
            response_format="json",
        )

        raw_hits = _parse_rerank_response(text)

        # Build a slug→candidate map for fast lookup
        cand_map = {c.slug: c for c in candidates}

        results: list[SearchHit] = []
        for item in raw_hits:
            if not isinstance(item, dict):
                continue
            slug = str(item.get("slug", ""))
            if slug not in cand_map:
                # Hallucinated slug — ignore (LW-12 defence)
                logger.debug("search_hallucinated_slug", slug=slug, file_id=file_id)
                continue
            try:
                rerank_score = float(item.get("rerank_score", 0.0))
            except (TypeError, ValueError):
                rerank_score = 0.0
            reason = str(item.get("reason", "")) or None
            base = cand_map[slug]
            results.append(
                SearchHit(
                    slug=base.slug,
                    title=base.title,
                    section=base.section,
                    similarity=base.similarity,
                    rerank_score=rerank_score,
                    reason=reason,
                )
            )

        results.sort(
            key=lambda h: (h.rerank_score or 0.0, h.similarity),
            reverse=True,
        )
        return results[: settings.search_final_k_max]


# ---------------------------------------------------------------------------
# Standalone parsing helper (also used in tests)
# ---------------------------------------------------------------------------


def _parse_rerank_response(raw: str) -> list[dict]:  # type: ignore[type-arg]
    """Parse the LLM re-rank JSON response defensively.

    Accepts:
    - ``{"hits": [...]}`` — canonical
    - ``{"candidates": [...]}`` / ``{"results": [...]}`` — aliases
    - ``[...]`` — bare array
    - Markdown-fenced blocks

    Args:
        raw: Raw string returned by the LLM.

    Returns:
        List of raw hit dicts.

    Raises:
        ValueError: If *raw* is not valid JSON or contains no extractable list.
    """
    text = raw.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        parts = text.split("```")
        inner = parts[1] if len(parts) > 1 else text
        if inner.startswith("json"):
            inner = inner[4:]
        text = inner.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Search Agent: invalid JSON from LLM: {exc}") from exc

    if isinstance(data, list):
        return data  # type: ignore[return-value]

    if isinstance(data, dict):
        for key in ("hits", "candidates", "results", "items", "matches", "pages"):
            value = data.get(key)
            if isinstance(value, list):
                return value  # type: ignore[return-value]
        if "slug" in data:
            return [data]  # type: ignore[return-value]

    raise ValueError(
        f"Search Agent: cannot extract hits array from LLM response: {raw[:300]!r}"
    )
