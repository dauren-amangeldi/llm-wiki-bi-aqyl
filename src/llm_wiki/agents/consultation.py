"""AI-advisor discovery step: classify the decision and either ask
clarification questions or signal that context is already sufficient.

Pure function — no DB, no HTTP. Called from the /consultations start
endpoint, which owns persistence.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import structlog

from llm_wiki.api.schemas import ClarificationQuestion
from llm_wiki.llm.chunk_store import ChunkHit, ChunkStore
from llm_wiki.llm.client import LLMClient

logger = structlog.get_logger(__name__)

MAX_QUESTIONS = 5
DISCOVERY_TOP_K = 5
DISCOVERY_FILE_ID = "advisor-discovery"

DISCOVERY_SYSTEM_PROMPT = """\
Ты — модуль постановки задачи AI-советника BI Group. По запросу руководителя:
1. Определи тип управленческого решения (initiative_scaling, investment,
   portfolio_prioritization, market_entry, org_change, cost_optimization,
   partner_or_tech_choice, risk_response, project_continuation).
2. Реши, достаточно ли контекста для рекомендации без уточнений.
3. Если недостаточно — сформулируй 1-5 вопросов (обычно 2-3), каждый из
   которых способен изменить итоговую рекомендацию. Каждый вопрос:
   короткая формулировка, 2-4 варианта (если уместны), почему это важно,
   allow_custom=true, required=false.
Не задавай вопрос, если ответ на него уже известен из запроса или найденных
материалов. Дипломатично, без давления. Верни строго JSON:
{"decision_type": str, "sufficient_context": bool, "questions": [...]}
"""


@dataclass
class DiscoveryResult:
    decision_type: str
    sufficient_context: bool
    questions: list[ClarificationQuestion] = field(default_factory=list)


def _format_light_matches(hits: list[ChunkHit]) -> str:
    if not hits:
        return "(релевантных материалов не найдено)"
    return "\n".join(f"- {hit.title or hit.slug}" for hit in hits)


def _parse_discovery_response(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("discovery response is not a JSON object")
    return data


async def run_discovery(
    llm: LLMClient, chunk_store: ChunkStore, *, query: str, role: str, language: str
) -> DiscoveryResult:
    """Classify the decision and produce a clarification-question block (or none)."""
    try:
        light_matches = chunk_store.query(query, top_k=DISCOVERY_TOP_K)
    except Exception as exc:  # noqa: BLE001
        logger.warning("discovery_chunk_query_failed", error=str(exc))
        light_matches = []

    context_note = _format_light_matches(light_matches)

    text, _usage = await llm.complete(
        prompt=(
            f"Роль: {role}. Язык ответа: {language}.\n"
            f"Запрос: {query}\n\n"
            f"Известные материалы:\n{context_note}"
        ),
        system=DISCOVERY_SYSTEM_PROMPT,
        file_id=DISCOVERY_FILE_ID,
        agent_type="advisor",
        response_format="json",
    )
    raw = _parse_discovery_response(text)

    questions = [ClarificationQuestion(**q) for q in raw.get("questions", [])][:MAX_QUESTIONS]
    return DiscoveryResult(
        decision_type=raw["decision_type"],
        sufficient_context=bool(raw.get("sufficient_context", False)),
        questions=questions,
    )
