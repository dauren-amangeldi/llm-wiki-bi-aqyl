"""AnswerAgent — RAG-style Q&A over the wiki (LW-20).

Pipeline:
    1. Embedding retrieval (heading-only collection from LW-11).
    2. If top_1 similarity < KEYWORD_FALLBACK_THRESHOLD, run a lightweight
       keyword scan over full page bodies and merge candidates.
    3. Load full page contents for the top-K survivors, truncate each to
       MAX_PAGE_CHARS to keep the LLM context bounded.
    4. Send to the LLM with a strict no-hallucination prompt.
    5. Parse the JSON response, validate cited slugs against the provided
       sources, return the structured answer + cost.

Refusal modes (no LLM call made — saves money):
    - top_k retrieval empty (ChromaDB has zero entries).
    - All retrieval candidates below NO_LLM_THRESHOLD (e.g., 0.30) AND the
      keyword fallback also finds nothing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import structlog

from llm_wiki.agents.base import BaseAgent
from llm_wiki.config import settings
from llm_wiki.llm.client import LLMClient, LLMUsage
from llm_wiki.llm.embeddings import EmbeddingStore, SearchHit

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Tunables — picked conservatively for the v1 heading-only index
# ---------------------------------------------------------------------------

NO_LLM_THRESHOLD: float = 0.30         # if best retrieval below this AND keyword fails, refuse
KEYWORD_FALLBACK_THRESHOLD: float = 0.45  # if best retrieval below this, run keyword scan
MAX_PAGE_CHARS: int = 4000              # truncate each page body in the assembled context
MAX_TOTAL_CONTEXT_CHARS: int = 24_000   # hard cap on total LLM input context

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
    ) -> None:
        self._llm = llm_client
        self._store = embedding_store
        self._wiki_dir = wiki_dir if wiki_dir is not None else settings.wiki_dir

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
            top_k: Maximum pages to feed the LLM (1–10).
            file_id: Correlation ID; default ``"ask"`` since there is no source file.

        Returns:
            ``AnswerResult`` with answer text, confidence, used sources, and total cost.
        """
        # ── Stage 1: embedding retrieval ─────────────────────────────────────
        try:
            candidates = self._store.query(question, top_k=top_k, file_id=file_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ask_embedding_failed", error=str(exc))
            candidates = []

        # ── Stage 2: keyword fallback if recall looks weak ───────────────────
        best_sim = max((c.similarity for c in candidates), default=0.0)
        if best_sim < KEYWORD_FALLBACK_THRESHOLD:
            extra = self._keyword_fallback(
                question, exclude={c.slug for c in candidates}, limit=top_k
            )
            candidates = (candidates + extra)[:top_k]

        # ── Refusal: nothing usable — do NOT call the LLM ───────────────────
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

        # ── Stage 3: load full page bodies and assemble context ──────────────
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

        # ── Stage 4: LLM call ─────────────────────────────────────────────────
        text, usage = await self._llm.complete(
            prompt=prompt,
            system="You are a precise wiki Q&A assistant. Return only valid JSON.",
            file_id=file_id,
            agent_type="answer",
            response_format="json",
        )

        # ── Stage 5: parse and validate ───────────────────────────────────────
        provided_slugs = {hit.slug for hit, _ in loaded}
        parsed = self._parse_response(text, provided_slugs)
        used_slugs = set(parsed["used_sources"])

        used_sources = [hit for hit, _ in loaded if hit.slug in used_slugs]
        # Transparency fallback: if the LLM reported no used sources, surface
        # the top retrieval hits anyway so the caller can show them.
        if not used_sources:
            used_sources = [hit for hit, _ in loaded[:3]]

        return AnswerResult(
            answer=parsed["answer"],
            confidence=parsed["confidence"],
            sources=used_sources,
            cost_usd=usage.cost_usd,
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _no_data_message(self) -> str:
        lang = settings.wiki_language.lower()
        if lang.startswith("kk"):
            return "Бұл сұраққа уикиде дерек жоқ."
        if lang.startswith("ru"):
            return "В вики нет данных по этому вопросу."
        return "The wiki does not contain information that answers this question."

    def _load_page_body(self, slug: str) -> str:
        path = self._wiki_dir / f"{slug}.md"
        if not path.exists():
            return ""
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("ask_load_page_failed", slug=slug, error=str(exc))
            return ""

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
        if not self._wiki_dir.exists():
            return []

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
        for path in self._wiki_dir.glob("*.md"):
            if path.stem in exclude:
                continue
            try:
                body = path.read_text(encoding="utf-8").lower()
            except OSError:
                continue
            score = sum(1 for n in needles if n in body)
            if score == 0:
                continue
            # Extract title from the first H1 if present; fall back to slug
            title = path.stem.replace("-", " ").title()
            first_line = body.splitlines()[0] if body else ""
            if first_line.startswith("# "):
                title = first_line[2:].strip()
            scored.append((score, path.stem, title))

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
