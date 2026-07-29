"""AdvisorAgent — semantic retrieval + structured insight synthesis (LW-N7 / LW-N9).

Pure business logic: receives query/role/language, returns dataclasses.
No FastAPI, Celery, or direct file I/O.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

import structlog

from llm_wiki.agents.answer import NO_LLM_THRESHOLD
from llm_wiki.agents.base import BaseAgent
from llm_wiki.llm.chunk_store import ChunkHit, ChunkStore
from llm_wiki.llm.client import LLMClient

logger = structlog.get_logger(__name__)

ADVISOR_TOP_K: int = 8

DEFAULT_SYSTEM_PROMPT: str = (
    "You are a precise knowledge-base advisor. "
    "Return only valid JSON matching the requested schema."
)


@dataclass(frozen=True)
class AdvisorPoint:
    """A single actionable insight in the advisor response."""

    heading: str
    body: str
    metric: str
    tag: str
    case_id: str


@dataclass(frozen=True)
class AdvisorResponse:
    """Structured advisor output matching the frontend contract."""

    title: str = ""
    summary: str = ""
    points: list[AdvisorPoint] = field(default_factory=list)
    source: str = ""
    caseCount: int = 0
    refusal: bool = False
    refusal_message: str = ""
    cost_usd: float = 0.0


@dataclass(frozen=True)
class HistoryTurn:
    """One turn in an advisor follow-up conversation."""

    role: Literal["user", "assistant"]
    content: str


class AdvisorAgent(BaseAgent):
    """Finds relevant cases via chunk retrieval and synthesises structured insights."""

    def __init__(self, llm_client: LLMClient, chunk_store: ChunkStore) -> None:
        self._llm = llm_client
        self._chunk_store = chunk_store

    async def run(self, *args: Any, **kwargs: Any) -> AdvisorResponse:
        """Execute ``advise()`` — satisfies BaseAgent ABC."""
        return await self.advise(*args, **kwargs)

    async def advise(
        self,
        query: str,
        role: str = "employee",
        language: str = "ru",
        title: str = "",
        history: list[HistoryTurn] | None = None,
        file_id: str = "advisor",
        system_prompt: str | None = None,
    ) -> AdvisorResponse:
        """Retrieve top-K chunks, group by case, and synthesise a structured answer.

        Args:
            query: User question (3–1000 chars, validated upstream).
            role: User role for tone/audience (``employee``, ``pm``, etc.).
            language: Response language (``ru``, ``en``, or ``kk``).
            title: Caller's job title from the Keycloak ``title`` claim (e.g.
                "AI Инженер Senior"); tailors depth/framing. "" → generic persona.
            history: Optional prior conversation turns for follow-ups.
            file_id: Correlation ID for usage logging.
            system_prompt: Role-specific system message from the skills table.

        Returns:
            ``AdvisorResponse`` — either populated insights or a refusal payload.
        """
        try:
            hits = self._chunk_store.query(
                query, top_k=ADVISOR_TOP_K, usage_file_id=file_id
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("advisor_chunk_query_failed", error=str(exc))
            hits = []

        best_sim = max((h.similarity for h in hits), default=0.0)
        if not hits or best_sim < NO_LLM_THRESHOLD:
            logger.info(
                "advisor_refused_low_similarity",
                query_len=len(query),
                best_sim=best_sim,
            )
            return AdvisorResponse(
                refusal=True,
                refusal_message=self._refusal_message(language),
            )

        cases_block, allowed_ids, case_texts = self._build_cases_context(hits)
        history_block = self._format_history(history or [])
        title = (title or "").strip()
        title_note = (
            f"The user's job title is **{title}** — calibrate the depth, framing "
            "and terminology of your advice to that role and seniority.\n"
            if title
            else ""
        )

        prompt = self._llm.load_prompt(
            "advisor",
            language=language,
            role=role,
            title_note=title_note,
            query=query,
            cases_block=cases_block,
            allowed_case_ids=", ".join(sorted(allowed_ids)) or "(none)",
            history_block=history_block,
        )

        text, usage = await self._llm.complete(
            prompt=prompt,
            system=system_prompt or DEFAULT_SYSTEM_PROMPT,
            file_id=file_id,
            agent_type="advisor",
            response_format="json",
        )

        parsed = self._parse_response(text, language)
        if parsed.get("refusal"):
            msg = str(parsed.get("refusal_message", "")).strip() or self._refusal_message(language)
            return AdvisorResponse(refusal=True, refusal_message=msg, cost_usd=usage.cost_usd)

        return self._build_response(parsed, allowed_ids, case_texts, hits, usage.cost_usd)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _refusal_message(self, language: str) -> str:
        lang = language.lower()
        if lang.startswith("kk"):
            return "Сұрауыңыз бойынша материалдар табылмады."
        if lang.startswith("en"):
            return "No materials found for your query."
        return "По вашему запросу материалов в базе знаний не найдено."

    def _build_cases_context(
        self, hits: list[ChunkHit]
    ) -> tuple[str, set[str], dict[str, str]]:
        """Group chunks by ``file_id`` and assemble the LLM context block."""
        by_case: dict[str, list[ChunkHit]] = {}
        for hit in hits:
            cid = hit.file_id or hit.slug
            by_case.setdefault(cid, []).append(hit)

        allowed_ids: set[str] = set(by_case)
        case_texts: dict[str, str] = {}
        blocks: list[str] = []

        for cid, chunks in by_case.items():
            chunks.sort(key=lambda c: c.similarity, reverse=True)
            title = chunks[0].title or cid
            parts: list[str] = []
            combined = ""
            for chunk in chunks:
                section = f" §{chunk.section}" if chunk.section else ""
                parts.append(
                    f"### [[{chunk.slug}]]{section} (sim={chunk.similarity:.2f})\n\n{chunk.text}"
                )
                combined += chunk.text + "\n"
            case_texts[cid] = combined
            blocks.append(f"## Case `{cid}` — {title}\n\n" + "\n\n".join(parts))

        return "\n\n---\n\n".join(blocks), allowed_ids, case_texts

    def _format_history(self, history: list[HistoryTurn]) -> str:
        if not history:
            return ""
        lines = ["# Conversation history", ""]
        for turn in history[-6:]:
            label = "User" if turn.role == "user" else "Advisor"
            lines.append(f"**{label}:** {turn.content}")
        lines.append("")
        return "\n".join(lines)

    def _parse_response(self, raw: str, language: str) -> dict[str, Any]:
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("advisor_response_unparseable", raw=raw[:200])
            return {"refusal": True, "refusal_message": self._refusal_message(language)}
        if not isinstance(data, dict):
            return {"refusal": True, "refusal_message": self._refusal_message(language)}
        return data

    def _build_response(
        self,
        data: dict[str, Any],
        allowed_ids: set[str],
        case_texts: dict[str, str],
        hits: list[ChunkHit],
        cost_usd: float,
    ) -> AdvisorResponse:
        raw_points = data.get("points", [])
        if not isinstance(raw_points, list):
            raw_points = []

        points: list[AdvisorPoint] = []
        for item in raw_points:
            if not isinstance(item, dict):
                continue
            case_id = str(item.get("case_id", "")).strip()
            if case_id not in allowed_ids:
                continue
            metric = str(item.get("metric", "")).strip()
            metric = self._verify_metric(metric, case_texts.get(case_id, ""))
            points.append(
                AdvisorPoint(
                    heading=str(item.get("heading", "")).strip(),
                    body=str(item.get("body", "")).strip(),
                    metric=metric,
                    tag=str(item.get("tag", "")).strip(),
                    case_id=case_id,
                )
            )

        if not points:
            lang = str(data.get("language", "ru"))
            return AdvisorResponse(
                refusal=True,
                refusal_message=self._refusal_message(lang),
                cost_usd=cost_usd,
            )

        distinct_in_hits = {h.file_id or h.slug for h in hits}
        case_count = len(distinct_in_hits)

        return AdvisorResponse(
            title=str(data.get("title", "")).strip(),
            summary=str(data.get("summary", "")).strip(),
            points=points,
            source=str(data.get("source", "")).strip(),
            caseCount=case_count,
            cost_usd=cost_usd,
        )

    def _verify_metric(self, metric: str, source_text: str) -> str:
        """Drop metrics that cannot be found verbatim in the source excerpt."""
        if not metric:
            return ""
        normalized_source = source_text.lower()
        # Require at least one numeric token from the metric to appear in source
        numbers = re.findall(r"\d[\d\s.,%]*\d|\d+", metric)
        if numbers:
            for num in numbers:
                needle = num.replace(" ", "").lower()
                if needle and needle in normalized_source.replace(" ", ""):
                    return metric
            return ""
        # Non-numeric metric: require substantial substring match
        if len(metric) >= 4 and metric.lower() in normalized_source:
            return metric
        return ""
