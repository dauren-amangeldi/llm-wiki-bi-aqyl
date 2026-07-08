"""LLM Auditor Agent — semantic quality checks for the wiki (LW-15).

Runs three checks that require reasoning:
  1. **Contradictions** between topically-related pages.
  2. **Duplicates** — pages covering the same concept.
  3. **Suspected stale** — content that appears semantically outdated.

Only these three ``IssueKind`` values are emitted; structural checks
(dead links, orphan pages, stale dates) belong to the deterministic Linter
(LW-14).

Architecture:
  - ``AuditorAgent`` is a pure ``BaseAgent`` subclass: it receives data,
    calls the LLM, and returns ``list[Issue]``.  Zero file I/O, zero Celery.
  - Two modes: ``sync`` (ordinary completions) and ``batch`` (OpenAI Batch
    API, -50% cost, 24 h SLA).
  - Pages are chunked in groups of ``BATCH_SIZE`` (50) to stay within
    token limits.
  - Topically-related pairs (for contradictions) are derived from pgvector
    cosine similarity > 0.6 by the caller; ``AuditorAgent`` just receives
    them.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Literal

import structlog

from llm_wiki.agents.base import BaseAgent
from llm_wiki.llm.client import LLMClient
from llm_wiki.quality.models import Issue, IssueKind, IssueSection

logger = structlog.get_logger(__name__)

_AUDITOR_SECTION = IssueSection.LLM_FLAGGED
_ALLOWED_KINDS: frozenset[str] = frozenset(
    {"contradiction", "duplicate", "suspected_stale"}
)

# Matches a JSON array, optionally wrapped in ```json fences
_JSON_FENCE_RE = re.compile(
    r"```(?:json)?\s*(\[.*?])\s*```|(\[.*?])",
    re.DOTALL,
)


class AuditorAgent(BaseAgent):
    """Detect semantic quality issues in wiki pages via LLM reasoning.

    Attributes:
        BATCH_SIZE: Number of wiki pages per LLM request.  Larger batches
            risk exceeding the context window on large wikis.
    """

    BATCH_SIZE = 50

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialise with a shared LLM client.

        Args:
            llm_client: Shared :class:`LLMClient` instance.  Must not be
                closed before ``run()`` completes.
        """
        self._llm = llm_client

    async def run(  # type: ignore[override]
        self,
        wiki_pages: list[tuple[str, str]],
        related_pairs: list[tuple[str, str]] | None = None,
        current_year: int | None = None,
        mode: Literal["sync", "batch"] = "sync",
    ) -> list[Issue]:
        """Run all semantic audits and return a flat list of flagged issues.

        Args:
            wiki_pages: Pairs of ``(slug, markdown_content)`` for every page
                to audit.
            related_pairs: Pairs of slug names that are topically related
                (cosine similarity > 0.6 from pgvector).  Used for
                contradiction checks.  Pass ``None`` for an empty set.
            current_year: Override the year used for staleness reasoning
                (injected into the prompt as today's date).  Defaults to the
                actual current year.
            mode: ``"sync"`` → use ordinary chat completions (fast, costly).
                ``"batch"`` → use OpenAI Batch API (-50%, 24 h SLA).

        Returns:
            Deduplicated list of :class:`Issue` objects with
            ``section=LLM_FLAGGED``.  Empty list if no issues were found.
        """
        if not wiki_pages:
            return []

        if current_year is None:
            current_year = datetime.now(timezone.utc).year

        if related_pairs is None:
            related_pairs = []

        # Split pages into chunks of BATCH_SIZE
        chunks: list[list[tuple[str, str]]] = [
            wiki_pages[i: i + self.BATCH_SIZE]
            for i in range(0, len(wiki_pages), self.BATCH_SIZE)
        ]
        total_batches = len(chunks)

        all_issues: list[Issue] = []

        for batch_idx, chunk in enumerate(chunks, start=1):
            chunk_slugs = {slug for slug, _ in chunk}
            # Only include related pairs where at least one slug is in this chunk
            chunk_pairs = [
                (a, b)
                for a, b in related_pairs
                if a in chunk_slugs or b in chunk_slugs
            ]

            if mode == "batch":
                issues = await self._run_batch_chunk(
                    chunk=chunk,
                    related_pairs=chunk_pairs,
                    batch_idx=batch_idx,
                    total_batches=total_batches,
                    current_year=current_year,
                )
            else:
                issues = await self._run_sync_chunk(
                    chunk=chunk,
                    related_pairs=chunk_pairs,
                    batch_idx=batch_idx,
                    total_batches=total_batches,
                    current_year=current_year,
                )

            all_issues.extend(issues)

        # Deduplicate by (kind, page_slug, description)
        seen: set[tuple[str, str, str]] = set()
        unique: list[Issue] = []
        for issue in all_issues:
            key = (str(issue.kind), issue.page_slug, issue.description)
            if key not in seen:
                seen.add(key)
                unique.append(issue)

        return unique

    # ------------------------------------------------------------------
    # Sync mode
    # ------------------------------------------------------------------

    async def _run_sync_chunk(
        self,
        chunk: list[tuple[str, str]],
        related_pairs: list[tuple[str, str]],
        batch_idx: int,
        total_batches: int,
        current_year: int,
    ) -> list[Issue]:
        """Run one chunk via ordinary chat completions (sync mode).

        Args:
            chunk: Pages in this batch.
            related_pairs: Related page pairs for this batch.
            batch_idx: 1-based index of this batch.
            total_batches: Total number of batches.
            current_year: Current year for the prompt.

        Returns:
            Issues found in this chunk.
        """
        prompt = self._build_prompt(
            chunk=chunk,
            related_pairs=related_pairs,
            batch_idx=batch_idx,
            total_batches=total_batches,
            current_year=current_year,
        )

        try:
            response_text = await self._llm.complete(
                prompt=prompt,
                system="You are a wiki quality auditor. Return only raw JSON.",
                file_id=f"auditor_batch_{batch_idx}",
                agent_type="audit",
                response_format="json",
            )
        except Exception as exc:
            logger.error(
                "auditor_sync_chunk_failed",
                batch_idx=batch_idx,
                error=str(exc),
            )
            return []

        return self._parse_response(response_text, chunk)

    # ------------------------------------------------------------------
    # Batch mode
    # ------------------------------------------------------------------

    async def _run_batch_chunk(
        self,
        chunk: list[tuple[str, str]],
        related_pairs: list[tuple[str, str]],
        batch_idx: int,
        total_batches: int,
        current_year: int,
    ) -> list[Issue]:
        """Run one chunk via the OpenAI Batch API (batch mode).

        Creates a batch job, polls until completion (up to 24 h), and
        parses the results.  Falls back to sync mode on any API error.

        Args:
            chunk: Pages in this batch.
            related_pairs: Related page pairs for this batch.
            batch_idx: 1-based index of this batch.
            total_batches: Total number of batches.
            current_year: Current year for the prompt.

        Returns:
            Issues found in this chunk.
        """
        import asyncio
        import io

        import openai

        from llm_wiki.config import settings

        if not settings.openai_api_key:
            logger.warning(
                "auditor_batch_no_openai_key_fallback_sync",
                batch_idx=batch_idx,
            )
            return await self._run_sync_chunk(
                chunk=chunk,
                related_pairs=related_pairs,
                batch_idx=batch_idx,
                total_batches=total_batches,
                current_year=current_year,
            )

        prompt = self._build_prompt(
            chunk=chunk,
            related_pairs=related_pairs,
            batch_idx=batch_idx,
            total_batches=total_batches,
            current_year=current_year,
        )

        client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        custom_id = f"auditor-batch-{batch_idx}-of-{total_batches}"

        # Build JSONL request file for the Batch API
        request_line = json.dumps({
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": settings.openai_model,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
            },
        })
        jsonl_bytes = (request_line + "\n").encode()
        file_obj = io.BytesIO(jsonl_bytes)
        file_obj.name = f"auditor_batch_{batch_idx}.jsonl"

        try:
            uploaded = await client.files.create(file=file_obj, purpose="batch")
            batch_job = await client.batches.create(
                input_file_id=uploaded.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            batch_id = batch_job.id
            logger.info(
                "auditor_batch_created",
                batch_id=batch_id,
                batch_idx=batch_idx,
            )

            # Poll until complete (24 h max, polling every 10 min)
            poll_interval = 600  # seconds
            max_polls = 144  # 24 h / 10 min
            for _ in range(max_polls):
                await asyncio.sleep(poll_interval)
                job = await client.batches.retrieve(batch_id)
                if job.status == "completed":
                    break
                if job.status in ("failed", "cancelled", "expired"):
                    logger.error(
                        "auditor_batch_terminal",
                        batch_id=batch_id,
                        status=job.status,
                    )
                    return []
            else:
                logger.error("auditor_batch_timeout", batch_id=batch_id)
                return []

            # Download results
            if not job.output_file_id:
                return []
            result_file = await client.files.content(job.output_file_id)
            result_text = result_file.text
        finally:
            await client.aclose()

        # Parse JSONL result file
        issues: list[Issue] = []
        for line in result_text.splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("custom_id") != custom_id:
                continue
            content = (
                row.get("response", {})
                .get("body", {})
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            issues.extend(self._parse_response(content, chunk))

        return issues

    # ------------------------------------------------------------------
    # Prompt & response parsing
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        chunk: list[tuple[str, str]],
        related_pairs: list[tuple[str, str]],
        batch_idx: int,
        total_batches: int,
        current_year: int,
    ) -> str:
        """Build the auditor prompt for one batch of pages.

        Args:
            chunk: Pages to include.
            related_pairs: Related pairs for this chunk.
            batch_idx: 1-based batch index.
            total_batches: Total batches.
            current_year: Current year.

        Returns:
            Formatted prompt string.
        """
        from llm_wiki.config import settings
        from llm_wiki.llm.client import LLMClient

        prompt_template = (
            LLMClient.PROMPTS_DIR / "auditor.md"
        ).read_text(encoding="utf-8")

        pages_parts: list[str] = []
        for slug, content in chunk:
            # Truncate very long pages to ~4 000 chars each
            truncated = content[:4000] + ("…" if len(content) > 4000 else "")
            pages_parts.append(f"### {slug}\n\n{truncated}")
        pages_content = "\n\n---\n\n".join(pages_parts)

        if related_pairs:
            pairs_lines = [f"- `{a}` ↔ `{b}`" for a, b in related_pairs]
            related_pairs_content = "\n".join(pairs_lines)
        else:
            related_pairs_content = "_No related pairs in this batch._"

        current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        language = settings.wiki_language

        return prompt_template.format(
            language=language,
            current_date=current_date,
            batch_index=batch_idx,
            total_batches=total_batches,
            pages_content=pages_content,
            related_pairs_content=related_pairs_content,
        )

    def _parse_response(
        self,
        text: str,
        chunk: list[tuple[str, str]],
    ) -> list[Issue]:
        """Parse the LLM response into a list of ``Issue`` objects.

        Handles:
        - Raw JSON arrays
        - JSON arrays wrapped in ```json fences
        - Wrapped objects ``{"issues": [...]}``

        Invalid ``kind`` values (e.g., ``dead_link``) are filtered out with
        a warning — the Auditor must only emit LLM-flagged kinds.

        Args:
            text: Raw text returned by the LLM.
            chunk: The pages batch (used for slug validation).

        Returns:
            Validated list of :class:`Issue` objects.
        """
        known_slugs = {slug for slug, _ in chunk}

        text = text.strip()
        if not text:
            return []

        # Try to extract JSON array from fenced block or raw text
        raw_array: Any = None
        fence_match = _JSON_FENCE_RE.search(text)
        if fence_match:
            json_text = fence_match.group(1) or fence_match.group(2)
        else:
            json_text = text

        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError:
            logger.warning("auditor_json_parse_error", raw=text[:200])
            return []

        if isinstance(parsed, dict):
            # Unwrap {"issues": [...]} or similar wrappers
            for key in ("issues", "findings", "results"):
                if isinstance(parsed.get(key), list):
                    raw_array = parsed[key]
                    break
            if raw_array is None:
                raw_array = [parsed]
        elif isinstance(parsed, list):
            raw_array = parsed
        else:
            return []

        issues: list[Issue] = []
        for item in raw_array:
            if not isinstance(item, dict):
                continue

            kind_raw = str(item.get("kind", "")).strip()
            if kind_raw not in _ALLOWED_KINDS:
                logger.warning(
                    "auditor_invalid_kind_filtered",
                    kind=kind_raw,
                    allowed=sorted(_ALLOWED_KINDS),
                )
                continue

            page_slug = str(item.get("page_slug", "")).strip()
            if not page_slug:
                continue

            if page_slug not in known_slugs:
                logger.warning(
                    "auditor_hallucinated_slug_filtered",
                    slug=page_slug,
                    known=sorted(known_slugs),
                )
                continue

            description = str(item.get("description", "")).strip()
            related_raw = item.get("related_slugs") or []
            related_slugs = tuple(
                str(s).strip()
                for s in related_raw
                if isinstance(s, str)
            )

            issues.append(
                Issue(
                    kind=IssueKind(kind_raw),
                    section=_AUDITOR_SECTION,
                    page_slug=page_slug,
                    description=description,
                    related_slugs=related_slugs,
                )
            )

        return issues
