"""AnswerAgent — RAG-style Q&A over the wiki (LW-20 / LW-20.1).

Two retrieval paths (selected by constructor argument):

ChunkStore path (LW-20.1, default when chunk_store is provided):
    1. Query ``ChunkStore`` for top-K chunks from page *bodies*.
    2. Assemble context directly from chunk text (no disk re-read needed).
    3. Send to LLM with no-hallucination prompt.
    4. Parse JSON, validate slug citations, return AnswerResult.

Heading-only fallback path (LW-20, used when chunk_store=None):
    1. Embedding retrieval from ``headings`` collection (page titles).
    2. Keyword fallback when best similarity < KEYWORD_FALLBACK_THRESHOLD.
    3. Load full page bodies from disk, truncate, assemble context.
    4–5. Same as above.

The chunk path has much better recall for questions whose answer is buried
in a page body with an unrelated title.  The heading path is kept as a
backward-compatible fallback (e.g., before reindex_chunks.py is first run).

Refusal modes (no LLM call — saves money):
    - ChunkStore path: collection is empty, or all chunks below NO_LLM_THRESHOLD.
    - Heading path: retrieval empty, or all below threshold AND keyword fails.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import structlog

from llm_wiki.agents.base import BaseAgent
from llm_wiki.config import settings
from llm_wiki.llm.chunk_store import ChunkHit, ChunkStore
from llm_wiki.llm.client import LLMClient
from llm_wiki.llm.embeddings import EmbeddingStore, SearchHit

if TYPE_CHECKING:
    from llm_wiki.storage.metadata import FileRecord

logger = structlog.get_logger(__name__)

# Strict Structured Outputs schema for the answer response. Enforced server-side
# by OpenAI (response_format=json_schema, strict=true) so the payload is
# guaranteed to be valid JSON with exactly these fields — no fence-stripping or
# best-effort parsing surprises. Every property must be listed in ``required``
# and ``additionalProperties`` must be false for strict mode to accept it.
_ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "used_sources": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer", "confidence", "used_sources"],
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# Tunables — picked conservatively for the v1 heading-only index
# ---------------------------------------------------------------------------

NO_LLM_THRESHOLD: float = 0.30         # if best retrieval below this AND keyword fails, refuse
KEYWORD_FALLBACK_THRESHOLD: float = 0.45  # if best retrieval below this, run keyword scan
MAX_PAGE_CHARS: int = 4000              # truncate each page body in the assembled context
MAX_TOTAL_CONTEXT_CHARS: int = 24_000   # hard cap on total LLM input context

_QA_SYSTEM_BASE = "You are a precise wiki Q&A assistant. Return only valid JSON."


def _qa_system(title: str = "") -> str:
    """System message for the Q&A call, optionally tailored to the caller's job
    title so the answer's depth and wording fit their role — without changing
    the facts or the required JSON structure. Empty title → the base persona."""
    title = (title or "").strip()
    if not title:
        return _QA_SYSTEM_BASE
    return (
        f'{_QA_SYSTEM_BASE} The person asking has the job title "{title}" — '
        "calibrate the depth and vocabulary of the answer text to that role, "
        "without changing the facts or the JSON structure."
    )

# ---------------------------------------------------------------------------
# Russian + Kazakh morphology aids (Patch 1)
# ---------------------------------------------------------------------------

_STOP_WORDS: frozenset[str] = frozenset({
    # Russian — high-frequency words that would otherwise dominate keyword scoring
    "что", "как", "это", "для", "при", "под", "над", "без", "или", "если",
    "чем", "так", "тот", "эта", "эти", "был", "была", "были", "быть",
    "его", "ее", "её", "их", "там", "тут", "где", "когда", "почему", "зачем",
    "кто", "какой", "какая", "какие", "какое", "какого", "какую",
    # Kazakh — equivalents
    "не", "немен", "қалай", "қандай", "қашан", "қайда", "осы", "сол",
    "бұл", "мен", "сен", "ол", "біз", "сіз", "олар", "үшін", "арқылы",
    # English connectors that may appear in mixed text
    "the", "and", "for", "with", "what", "how", "why", "when",
})

_PREFIX_LEN: int = 5  # first N chars used for prefix matching of long tokens


@dataclass(frozen=True)
class AnswerResult:
    """Structured outcome of ``AnswerAgent.answer()``."""

    answer: str
    confidence: Literal["high", "medium", "low"]
    sources: list[SearchHit]  # ordered by similarity desc
    cost_usd: float


class AnswerAgent(BaseAgent):
    """Synthesises Q&A answers from the wiki using retrieval + LLM."""

    def __init__(
        self,
        llm_client: LLMClient,
        embedding_store: EmbeddingStore,
        wiki_dir: Path | None = None,
        chunk_store: ChunkStore | None = None,
    ) -> None:
        self._llm = llm_client
        self._store = embedding_store
        self._wiki_dir = wiki_dir if wiki_dir is not None else settings.wiki_dir
        self._chunk_store = chunk_store

    async def run(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        raise NotImplementedError("Call answer() directly")

    async def answer(
        self,
        question: str,
        top_k: int = 5,
        file_id: str = "ask",
    ) -> AnswerResult:
        """Run the Q&A pipeline.

        Args:
            question: User question, 3–1000 chars (validated upstream).
            top_k: Maximum pages (heading path) or chunks (chunk path) to consider.
            file_id: Correlation ID; default ``"ask"`` since there is no source file.

        Returns:
            ``AnswerResult`` with answer text, confidence, used sources, and total cost.
        """
        if self._chunk_store is not None:
            return await self._answer_via_chunks(question, file_id=file_id)
        return await self._answer_via_headings(question, top_k=top_k, file_id=file_id)

    async def answer_for_document(
        self,
        question: str,
        document: FileRecord,
        language: str = "ru",
        file_id: str = "ask",
        title: str = "",
    ) -> AnswerResult:
        """Ask scoped to a specific uploaded document.

        Loads the wiki pages this document created/updated and feeds them
        directly to the LLM as guaranteed context — no similarity threshold,
        no refusal path. If the wiki content exists, we answer from it.

        Args:
            question: User question.
            document: FileRecord for the uploaded source file.
            language: Response language (``ru``, ``en``, or ``kk``).
            file_id: Correlation ID for usage logging.

        Returns:
            ``AnswerResult`` with answer text, confidence, sources, and cost.
        """
        slugs = list(
            dict.fromkeys(
                list(document.created_pages or []) + list(document.updated_pages or [])
            )
        )
        return await self._answer_from_slugs(
            question=question,
            slugs=slugs,
            language=language,
            file_id=file_id,
            status=document.status,
            title=title,
        )

    async def answer_for_case(
        self,
        question: str,
        documents: list[FileRecord],
        language: str = "ru",
        file_id: str = "ask",
        title: str = "",
    ) -> AnswerResult:
        """Ask scoped to a whole case — answers across all its documents.

        Aggregates the wiki pages created/updated by every document linked to
        the case and feeds them as one context block (NotebookLM-style Q&A over
        the full case rather than a single source).

        Args:
            question: User question.
            documents: FileRecords for all documents linked to the case.
            language: Response language (``ru``, ``en``, or ``kk``).
            file_id: Correlation ID for usage logging.

        Returns:
            ``AnswerResult`` with answer text, confidence, sources, and cost.
        """
        slugs: list[str] = []
        for doc in documents:
            slugs.extend(list(doc.created_pages or []))
            slugs.extend(list(doc.updated_pages or []))
        slugs = list(dict.fromkeys(slugs))

        # Surface the least-finished document status so the "still processing"
        # message is accurate when no pages exist yet.
        statuses = {d.status for d in documents}
        status_label = "EMPTY" if not documents else ", ".join(sorted(statuses))

        return await self._answer_from_slugs(
            question=question,
            slugs=slugs,
            language=language,
            file_id=file_id,
            status=status_label,
            title=title,
        )

    async def _answer_from_slugs(
        self,
        question: str,
        slugs: list[str],
        language: str,
        file_id: str,
        status: str,
        title: str = "",
    ) -> AnswerResult:
        """Answer a question from a fixed set of wiki page slugs (no retrieval).

        Shared core for ``answer_for_document`` and ``answer_for_case``: loads
        the given pages as guaranteed context and synthesises an answer.
        """
        loaded: list[tuple[str, str]] = []
        total = 0
        for slug in slugs:
            body = self._load_page_body(slug)
            if not body:
                continue
            truncated = body[:MAX_PAGE_CHARS]
            if total + len(truncated) > MAX_TOTAL_CONTEXT_CHARS:
                break
            loaded.append((slug, truncated))
            total += len(truncated)

        if not loaded:
            msg = {
                "ru": (
                    f"Файл ещё обрабатывается или wiki-страница не создана "
                    f"(статус: {status})."
                ),
                "kk": (
                    f"Файл әлі өңделуде немесе уики-беті жасалмаған "
                    f"(статус: {status})."
                ),
                "en": (
                    f"The file is still being processed or no wiki page was created yet "
                    f"(status: {status})."
                ),
            }.get(
                language,
                (
                    f"The file is still being processed or no wiki page was created yet "
                    f"(status: {status})."
                ),
            )
            return AnswerResult(answer=msg, confidence="low", sources=[], cost_usd=0.0)

        sources_block = "\n\n---\n\n".join(
            f"## Source: [[{slug}]]\n\n{body}" for slug, body in loaded
        )
        prompt = self._llm.load_prompt(
            "answer",
            language=language,
            question=question,
            sources_block=sources_block,
        )
        text, usage = await self._llm.complete(
            prompt=prompt,
            system=_qa_system(title),
            file_id=file_id,
            agent_type="answer",
            response_format="json",
            json_schema=_ANSWER_SCHEMA,
            schema_name="wiki_answer",
        )

        provided_slugs = {slug for slug, _ in loaded}
        parsed = self._parse_response(text, provided_slugs)
        # Resolve each cited slug to its human page title so the frontend shows
        # "Деловой отчёт…" in the citation footer instead of a raw slug/UUID.
        used = [
            SearchHit(slug=s, title=self._load_page_title(s), section="", similarity=1.0)
            for s in parsed["used_sources"]
        ] or [
            SearchHit(slug=s, title=self._load_page_title(s), section="", similarity=1.0)
            for s, _ in loaded[:3]
        ]

        confidence: Literal["high", "medium", "low"] = parsed["confidence"]
        if confidence == "low":
            confidence = "medium"

        return AnswerResult(
            answer=parsed["answer"],
            confidence=confidence,
            sources=used,
            cost_usd=usage.cost_usd,
        )

    # ── Chunk-based retrieval path (LW-20.1) ─────────────────────────────────

    async def _answer_via_chunks(
        self, question: str, file_id: str
    ) -> AnswerResult:
        """RAG pipeline using the ``chunks`` collection (full page bodies)."""
        assert self._chunk_store is not None  # guarded by caller

        # Stage 1: chunk retrieval
        try:
            chunk_hits = self._chunk_store.query(
                question,
                top_k=settings.chunk_retrieval_top_k,
                usage_file_id=file_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("ask_chunk_query_failed", error=str(exc))
            chunk_hits = []

        # Refusal: nothing usable
        best_sim = max((c.similarity for c in chunk_hits), default=0.0)
        if not chunk_hits or best_sim < NO_LLM_THRESHOLD:
            logger.info(
                "ask_refused_no_chunks",
                question_len=len(question),
                best_sim=best_sim,
            )
            return AnswerResult(
                answer=self._no_data_message(),
                confidence="low",
                sources=[],
                cost_usd=0.0,
            )

        # Stage 2: assemble context directly from chunk text (no disk re-read)
        total_chars = 0
        loaded_chunks: list[ChunkHit] = []
        for chunk in chunk_hits:
            if total_chars + len(chunk.text) > MAX_TOTAL_CONTEXT_CHARS:
                break
            loaded_chunks.append(chunk)
            total_chars += len(chunk.text)

        if not loaded_chunks:
            return AnswerResult(
                answer=self._no_data_message(),
                confidence="low",
                sources=[],
                cost_usd=0.0,
            )

        # Multiple chunks from the same page are intentionally kept — they
        # increase the signal that the page is relevant.
        section_suffix = lambda c: f" §{c.section}" if c.section else ""  # noqa: E731
        sources_block = "\n\n---\n\n".join(
            f"## Source: [[{c.slug}]]{section_suffix(c)} (similarity={c.similarity:.2f})\n\n{c.text}"
            for c in loaded_chunks
        )

        prompt = self._llm.load_prompt(
            "answer",
            language=settings.wiki_language,
            question=question,
            sources_block=sources_block,
        )

        # Stage 3: LLM call
        text, usage = await self._llm.complete(
            prompt=prompt,
            system="You are a precise wiki Q&A assistant. Return only valid JSON.",
            file_id=file_id,
            agent_type="answer",
            response_format="json",
            json_schema=_ANSWER_SCHEMA,
            schema_name="wiki_answer",
        )

        # Stage 4: parse + validate
        provided_slugs = {c.slug for c in loaded_chunks}
        parsed = self._parse_response(text, provided_slugs)
        used_slugs = set(parsed["used_sources"])

        # Map ChunkHit → SearchHit for API compatibility; deduplicate by slug,
        # keeping the highest-similarity chunk per page.
        best_by_slug: dict[str, ChunkHit] = {}
        for chunk in loaded_chunks:
            existing = best_by_slug.get(chunk.slug)
            if existing is None or chunk.similarity > existing.similarity:
                best_by_slug[chunk.slug] = chunk

        used_sources: list[SearchHit] = [
            SearchHit(slug=c.slug, title=c.title, section=c.section, similarity=c.similarity)
            for slug, c in best_by_slug.items()
            if slug in used_slugs
        ]
        if not used_sources:
            # Transparency fallback — show top 3 unique pages
            seen: set[str] = set()
            for chunk in loaded_chunks:
                if chunk.slug not in seen:
                    used_sources.append(
                        SearchHit(
                            slug=chunk.slug,
                            title=chunk.title,
                            section=chunk.section,
                            similarity=chunk.similarity,
                        )
                    )
                    seen.add(chunk.slug)
                    if len(used_sources) == 3:
                        break

        return AnswerResult(
            answer=parsed["answer"],
            confidence=parsed["confidence"],
            sources=used_sources,
            cost_usd=usage.cost_usd,
        )

    # ── Heading-based retrieval path (LW-20, fallback) ───────────────────────

    async def _answer_via_headings(
        self, question: str, top_k: int, file_id: str
    ) -> AnswerResult:
        """Original RAG pipeline using the ``headings`` collection (page titles)."""
        # Stage 1: embedding retrieval
        try:
            candidates = self._store.query(question, top_k=top_k, file_id=file_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ask_embedding_failed", error=str(exc))
            candidates = []

        # Stage 2: keyword fallback if recall looks weak
        best_sim = max((c.similarity for c in candidates), default=0.0)
        if best_sim < KEYWORD_FALLBACK_THRESHOLD:
            extra = self._keyword_fallback(
                question, exclude={c.slug for c in candidates}, limit=top_k
            )
            candidates = (candidates + extra)[:top_k]

        # Refusal
        best_sim_after = max((c.similarity for c in candidates), default=0.0)
        has_keyword_hit = any(c.similarity == 1.0 for c in candidates)
        if not candidates or (best_sim_after < NO_LLM_THRESHOLD and not has_keyword_hit):
            logger.info(
                "ask_refused_no_candidates",
                question_len=len(question),
                best_sim=best_sim,
            )
            return AnswerResult(
                answer=self._no_data_message(),
                confidence="low",
                sources=[],
                cost_usd=0.0,
            )

        # Stage 3: load full page bodies
        loaded: list[tuple[SearchHit, str]] = []
        total_chars = 0
        for hit in candidates:
            body = self._load_page_body(hit.slug)
            if not body:
                continue
            truncated = body[:MAX_PAGE_CHARS]
            if total_chars + len(truncated) > MAX_TOTAL_CONTEXT_CHARS:
                break
            loaded.append((hit, truncated))
            total_chars += len(truncated)

        if not loaded:
            return AnswerResult(
                answer=self._no_data_message(),
                confidence="low",
                sources=[],
                cost_usd=0.0,
            )

        sources_block = "\n\n---\n\n".join(
            f"## Source: [[{hit.slug}]] (similarity={hit.similarity:.2f})\n\n{body}"
            for hit, body in loaded
        )

        prompt = self._llm.load_prompt(
            "answer",
            language=settings.wiki_language,
            question=question,
            sources_block=sources_block,
        )

        # Stage 4: LLM call
        text, usage = await self._llm.complete(
            prompt=prompt,
            system="You are a precise wiki Q&A assistant. Return only valid JSON.",
            file_id=file_id,
            agent_type="answer",
            response_format="json",
            json_schema=_ANSWER_SCHEMA,
            schema_name="wiki_answer",
        )

        # Stage 5: parse and validate
        provided_slugs = {hit.slug for hit, _ in loaded}
        parsed = self._parse_response(text, provided_slugs)
        used_slugs = set(parsed["used_sources"])

        used_sources = [hit for hit, _ in loaded if hit.slug in used_slugs]
        if not used_sources:
            used_sources = [hit for hit, _ in loaded[:3]]

        return AnswerResult(
            answer=parsed["answer"],
            confidence=parsed["confidence"],
            sources=used_sources,
            cost_usd=usage.cost_usd,
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _no_data_message(self, language: str | None = None) -> str:
        lang = (language or settings.wiki_language).lower()
        if lang.startswith("kk"):
            return "Бұл сұраққа уикиде дерек жоқ."
        if lang.startswith("ru"):
            return "В вики нет данных по этому вопросу."
        return "The wiki does not contain information that answers this question."

    def _load_page_body(self, slug: str) -> str:
        from llm_wiki.storage import wiki_store

        try:
            return wiki_store.get_page(slug) or ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("ask_load_page_failed", slug=slug, error=str(exc))
            return ""

    def _load_page_title(self, slug: str) -> str:
        """Human page title for a slug (the stored ``wiki_fts.title``), falling
        back to a humanised slug when the page has no title or isn't found —
        so a citation never surfaces a raw slug/UUID to the reader."""
        from llm_wiki.storage import wiki_store

        try:
            title = wiki_store.get_page_title(slug)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ask_load_title_failed", slug=slug, error=str(exc))
            title = None
        return (title or "").strip() or slug.replace("-", " ").title()

    def _keyword_fallback(
        self, question: str, exclude: set[str], limit: int
    ) -> list[SearchHit]:
        """Lightweight keyword scan with stop-word filtering + prefix matching.

        Russian and Kazakh have rich morphology — pure substring match is too
        strict (``оптимизаторов`` would miss ``оптимизатор``).  We compensate
        by prefix-matching long tokens (first ``_PREFIX_LEN`` chars), which is
        morphology-naive but cheap and reasonably effective on Cyrillic.

        Keyword hits are returned with ``similarity=1.0`` to signal an exact
        match — this score is treated as a "present" marker and prevents the
        NO_LLM_THRESHOLD refusal path even when embedding recall is weak.
        """
        from llm_wiki.storage import wiki_store

        raw_tokens = re.findall(r"[a-z\u0400-\u04ff0-9]{3,}", question.lower())
        tokens = [t for t in raw_tokens if t not in _STOP_WORDS]
        if not tokens:
            return []

        # Long tokens (≥6 chars): use prefix; short tokens: exact substring
        needles: list[str] = [
            t[:_PREFIX_LEN] if len(t) >= 6 else t
            for t in tokens
        ]
        # Deduplicate while preserving order
        needles = list(dict.fromkeys(needles))

        scored: list[tuple[int, str, str]] = []
        for slug, raw in wiki_store.get_all_pages():
            if slug in exclude:
                continue
            body = raw.lower()
            score = sum(1 for n in needles if n in body)
            if score == 0:
                continue
            # Extract title from the first H1 if present; fall back to slug
            title = slug.replace("-", " ").title()
            first_line = body.splitlines()[0] if body else ""
            if first_line.startswith("# "):
                title = first_line[2:].strip()
            scored.append((score, slug, title))

        scored.sort(reverse=True)
        return [
            SearchHit(slug=slug, title=title, section="", similarity=1.0)
            for _score, slug, title in scored[:limit]
        ]

    def _parse_response(
        self, raw: str, provided_slugs: set[str]
    ) -> dict[str, Any]:
        """Parse and validate the LLM's JSON answer.

        Strips markdown fences if present.  Falls back to a low-confidence
        refusal payload on any parsing error so the caller always receives a
        well-formed result.
        """
        text = raw.strip()
        # Strip optional ```json ... ``` fences
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("ask_response_unparseable", raw=raw[:200])
            return {
                "answer": self._no_data_message(),
                "confidence": "low",
                "used_sources": [],
            }

        answer = str(data.get("answer", "")).strip() or self._no_data_message()
        confidence = data.get("confidence", "low")
        if confidence not in ("high", "medium", "low"):
            confidence = "low"

        used = data.get("used_sources", [])
        if not isinstance(used, list):
            used = []
        # Filter out hallucinated slugs
        used_valid = [s for s in used if isinstance(s, str) and s in provided_slugs]

        # Scrub hallucinated [[slug]] citations from the answer body, replacing
        # them with the plain slug text so the answer stays readable.
        def _scrub(match: re.Match[str]) -> str:
            slug = match.group(1)
            return f"[[{slug}]]" if slug in provided_slugs else slug

        answer = re.sub(r"\[\[([a-z0-9][a-z0-9-]*[a-z0-9]?)\]\]", _scrub, answer)

        return {"answer": answer, "confidence": confidence, "used_sources": used_valid}
