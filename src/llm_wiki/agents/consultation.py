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
from pydantic import ValidationError

from llm_wiki.api.schemas import ClarificationQuestion, DecisionBrief, QuestionAnswer, UnderstandingSnapshot
from llm_wiki.llm.chunk_store import ChunkHit, ChunkStore
from llm_wiki.llm.client import LLMClient

logger = structlog.get_logger(__name__)

MAX_QUESTIONS = 5
DISCOVERY_TOP_K = 5
SYNTHESIS_TOP_K = 10
DISCOVERY_FILE_ID = "advisor-discovery"
SNAPSHOT_FILE_ID = "advisor-snapshot"
SYNTHESIS_FILE_ID = "advisor-synthesis"
CRITIQUE_FILE_ID = "advisor-critique"

SYNTHESIS_SYSTEM_PROMPT = """\
Ты формируешь decision brief для руководителя BI Group на основе снимка
понимания и найденных внутренних кейсов. Раздели факты, вводные
пользователя, свои выводы и допущения. Рекомендация — первой. Верни строго
JSON с полями: recommendation, why_now, problem_frame, key_assumption,
rationale, alternatives (list[str]), risks (list[str]), first_step,
reconsider_if (list[str]), evidence_strength ("high"/"medium"/"low"),
assumptions (list[str]), sources (list[str]).
"""

CRITIQUE_SYSTEM_PROMPT = """\
Ты — внутренний критик уже сформированного decision brief. Проверь: слабое
допущение, альтернативное объяснение, опровергающие кейсы, последствия
второго порядка, соответствие источникам. Верни JSON только с полями,
которые нужно скорректировать по итогам критики: risks (list[str]),
evidence_strength ("high"/"medium"/"low"), reconsider_if (list[str]).
Не смягчай тон искусственно — только по существу.
"""

SNAPSHOT_SYSTEM_PROMPT = """\
Ты собираешь снимок понимания управленческого решения из исходного запроса
и ответов на уточняющие вопросы (пропущенные вопросы — это отсутствие
данных, а не отказ от решения). Верни строго JSON с полями: decision,
desired_outcome, horizon, constraints (list[str]), stakeholders (list[str]),
success_criteria (list[str]), assumptions (list[str]). Не выдумывай факты —
если данных нет, оставляй список пустым или допущение явным.
"""

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


def _parse_json_object(raw: str, *, log_event: str) -> dict[str, Any] | None:
    """Parse an LLM JSON reply, returning ``None`` on any malformed input.

    Mirrors the graceful-degradation pattern used by
    ``AdvisorAgent._parse_response`` (advisor.py:194-206): callers get a
    sentinel instead of a propagated exception, and log the raw text for
    debugging. Shared by the discovery and understanding-snapshot steps.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"{log_event}_unparseable", raw=raw[:200])
        return None
    if not isinstance(data, dict):
        logger.warning(f"{log_event}_not_object", raw=raw[:200])
        return None
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
    raw = _parse_json_object(text, log_event="discovery_response")
    if raw is None or "decision_type" not in raw:
        logger.warning("discovery_response_invalid", raw=text[:200])
        return DiscoveryResult(decision_type="unknown", sufficient_context=False, questions=[])

    raw_questions = raw.get("questions", [])
    if not isinstance(raw_questions, list):
        logger.warning("discovery_questions_not_list", questions=raw_questions)
        raw_questions = []

    questions: list[ClarificationQuestion] = []
    for item in raw_questions[:MAX_QUESTIONS]:
        try:
            questions.append(ClarificationQuestion(**item))
        except (TypeError, ValidationError) as exc:
            logger.warning("discovery_question_invalid", error=str(exc), item=item)

    return DiscoveryResult(
        decision_type=str(raw["decision_type"]),
        sufficient_context=bool(raw.get("sufficient_context", False)),
        questions=questions,
    )


def _fallback_snapshot(query: str) -> UnderstandingSnapshot:
    """Degrade gracefully when the LLM reply is missing or malformed: keep the
    original query as the decision and leave the rest for a human to fill in."""
    return UnderstandingSnapshot(
        decision=query,
        desired_outcome="",
        horizon="",
        constraints=[],
        stakeholders=[],
        success_criteria=[],
        assumptions=[],
    )


def _format_qa(questions: list[ClarificationQuestion], answers: list[QuestionAnswer]) -> str:
    if not answers:
        return "(вопросы не задавались или все пропущены)"
    qa_by_id = {q.id: q for q in questions}
    lines = []
    for a in answers:
        q = qa_by_id.get(a.question_id)
        q_text = q.text if q else a.question_id
        lines.append(f"- {q_text} → {'(пропущено)' if a.skipped else a.answer}")
    return "\n".join(lines)


async def build_snapshot(
    llm: LLMClient, *, query: str, questions: list[ClarificationQuestion], answers: list[QuestionAnswer]
) -> UnderstandingSnapshot:
    """Собрать снимок понимания из исходного запроса и ответов пользователя.

    Graceful degradation mirrors ``run_discovery``: any malformed or
    unparseable LLM reply falls back to a blank snapshot instead of raising.
    """
    qa_block = _format_qa(questions, answers)

    text, _usage = await llm.complete(
        prompt=f"Исходный запрос: {query}\n\nОтветы:\n{qa_block}",
        system=SNAPSHOT_SYSTEM_PROMPT,
        file_id=SNAPSHOT_FILE_ID,
        agent_type="advisor",
        response_format="json",
    )
    raw = _parse_json_object(text, log_event="snapshot_response")
    if raw is None:
        logger.warning("snapshot_response_invalid", raw=text[:200])
        return _fallback_snapshot(query)

    try:
        return UnderstandingSnapshot(**raw)
    except ValidationError as exc:
        logger.warning("snapshot_response_validation_failed", error=str(exc), raw=text[:200])
        return _fallback_snapshot(query)


def _format_deep_matches(hits: list[ChunkHit]) -> str:
    if not hits:
        return "(релевантных кейсов не найдено)"
    return "\n".join(f"- {hit.title or hit.slug}: {hit.text}" for hit in hits)


def _fallback_brief(query: str) -> DecisionBrief:
    """Degrade gracefully when the draft LLM reply is missing or malformed: keep
    the brief minimally honest instead of raising out of a pure function."""
    return DecisionBrief(
        recommendation=f"Недостаточно данных для рекомендации по запросу: {query}",
        why_now="",
        problem_frame=query,
        key_assumption="",
        rationale="Черновик decision brief не удалось сформировать из ответа модели.",
        alternatives=[],
        risks=[],
        first_step="Уточнить запрос и повторить синтез вручную.",
        reconsider_if=[],
        evidence_strength="low",
        assumptions=[],
        sources=[],
    )


async def run_synthesis(
    llm: LLMClient, chunk_store: ChunkStore, *, query: str, snapshot: UnderstandingSnapshot, language: str
) -> DecisionBrief:
    """Глубокий поиск → синтез рекомендации → внутренний критический проход.

    Graceful degradation mirrors ``run_discovery``/``build_snapshot``: a
    malformed draft reply falls back to a minimal low-confidence brief; a
    malformed critique reply simply keeps the draft brief unchanged instead
    of failing the whole synthesis.
    """
    try:
        deep_matches = chunk_store.query(f"{query}\n{snapshot.decision}", top_k=SYNTHESIS_TOP_K)
    except Exception as exc:  # noqa: BLE001
        logger.warning("synthesis_chunk_query_failed", error=str(exc))
        deep_matches = []

    context_block = _format_deep_matches(deep_matches)

    draft_text, _usage = await llm.complete(
        prompt=(
            f"Язык ответа: {language}.\nСнимок понимания: {snapshot.model_dump_json()}\n\n"
            f"Найденные материалы:\n{context_block}"
        ),
        system=SYNTHESIS_SYSTEM_PROMPT,
        file_id=SYNTHESIS_FILE_ID,
        agent_type="advisor",
        response_format="json",
    )
    draft_raw = _parse_json_object(draft_text, log_event="synthesis_draft_response")
    if draft_raw is None:
        logger.warning("synthesis_draft_response_invalid", raw=draft_text[:200])
        return _fallback_brief(query)

    try:
        draft = DecisionBrief(**draft_raw)
    except (TypeError, ValidationError) as exc:
        logger.warning("synthesis_draft_response_validation_failed", error=str(exc), raw=draft_text[:200])
        return _fallback_brief(query)

    critique_text, _usage = await llm.complete(
        prompt=f"Черновик brief: {draft.model_dump_json()}\n\nНайденные материалы:\n{context_block}",
        system=CRITIQUE_SYSTEM_PROMPT,
        file_id=CRITIQUE_FILE_ID,
        agent_type="advisor",
        response_format="json",
    )
    critique_raw = _parse_json_object(critique_text, log_event="synthesis_critique_response")
    if critique_raw is None:
        logger.warning("synthesis_critique_response_invalid", raw=critique_text[:200])
        return draft

    return draft.model_copy(update={k: v for k, v in critique_raw.items() if v is not None})
