"""Персистентные консультации AI-советника (BUG-03).

Сервер ведёт строку консультации по шагам живого флоу фронта
(stores/advisorFlow.ts): создание генерирует уточняющие вопросы, ответы
прогоняются через «как я понял», финальный бриф присылает клиент после
SSE-события /advisor. Всё владельческое: чужая консультация = 404 (как твины).

Заодно закрывает часть BUG-16: GET /advisor/consultations больше не 404.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from fastapi import Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.api.deps import get_db, get_user_key
from llm_wiki.api.v1 import router
from llm_wiki.storage.metadata import AdvisorConsultation

logger = structlog.get_logger(__name__)


async def _require_own(
    db: AsyncSession, consultation_id: str, caller: str
) -> AdvisorConsultation:
    row = await db.get(AdvisorConsultation, consultation_id)
    if row is None or row.owner != caller:
        # 404, не 403 — не подтверждаем существование чужой консультации.
        raise HTTPException(status_code=404, detail="Consultation not found")
    return row


def _serialize(row: AdvisorConsultation) -> dict[str, Any]:
    return {
        "id": row.id,
        "title": row.title,
        "situation": row.situation,
        "language": row.language,
        "step": row.step,
        "decision_type_label": row.decision_type_label,
        "questions": row.questions or [],
        "answers": row.answers or {},
        "understanding": row.understanding,
        "brief": row.brief,
        "outcome": row.outcome,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


class StartBody(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    language: str = "ru"


@router.post("/advisor/consultations", status_code=201)
async def start_consultation(
    body: StartBody,
    db: AsyncSession = Depends(get_db),
    caller: str = Depends(get_user_key),
) -> dict[str, Any]:
    """Создать консультацию и подобрать уточняющие вопросы (один вызов).

    Замена клиентского POST /advisor/questions: та же генерация, но результат
    сразу сохранён — F5 на шаге вопросов больше ничего не теряет.
    """
    from llm_wiki.agents.advisor_questions import (
        DECISION_TYPE_LABELS,
        generate_questions,
    )
    from llm_wiki.llm.client import LLMClient

    llm = LLMClient()
    try:
        result = await generate_questions(llm, body.query, body.language)
    finally:
        await llm.aclose()

    decision_type = result["decision_type"]
    label = DECISION_TYPE_LABELS.get(decision_type, DECISION_TYPE_LABELS["generic"])
    row = AdvisorConsultation(
        id=uuid.uuid4().hex,
        owner=caller,
        title=body.query.strip()[:100],
        situation=body.query.strip(),
        language=body.language,
        step="questions",
        decision_type_label=label,
        questions=list(result["questions"]),
    )
    db.add(row)
    await db.commit()
    logger.info("advisor_consultation_started", consultation_id=row.id)
    return {
        "id": row.id,
        "decision_type": decision_type,
        "decision_type_label": label,
        "questions": row.questions,
    }


class AnswerItem(BaseModel):
    question: str = ""
    answer: str = ""


class AnswersBody(BaseModel):
    # Ответы двумя видами: для LLM-пересказа (текст вопроса + текст ответа)
    # и сырой стейт фронта {question_id: [...]} — чтобы восстановить галочки.
    answers: list[AnswerItem] = Field(default_factory=list)
    raw_answers: dict[str, list[str]] = Field(default_factory=dict)


@router.post("/advisor/consultations/{consultation_id}/answers")
async def submit_answers(
    consultation_id: str,
    body: AnswersBody,
    db: AsyncSession = Depends(get_db),
    caller: str = Depends(get_user_key),
) -> dict[str, Any]:
    """Сохранить ответы и получить «как я понял» (замена /advisor/understand)."""
    import json

    from llm_wiki.llm.client import LLMClient

    row = await _require_own(db, consultation_id, caller)

    qa = "\n".join(f"- {a.question}: {a.answer}" for a in body.answers if a.answer.strip())
    prompt = (
        f"Respond in language: {row.language}. In 2–4 natural sentences, restate what "
        "you (the advisor) understood about the user's situation, weaving in their "
        "answers as a coherent summary — do NOT list them as question/answer pairs.\n\n"
        f"Situation:\n{row.situation}\n\nUser's answers:\n{qa or '(none)'}\n\n"
        'Return JSON: {"understanding": "<2-4 sentences>"}.'
    )
    understanding = row.situation
    llm = LLMClient()
    try:
        text, _usage = await llm.complete(
            prompt=prompt,
            system="You are a precise advisor. Return only valid JSON.",
            file_id=f"advisor-consultation-{consultation_id}",
            agent_type="advisor",
            response_format="json",
        )
        understanding = (
            str(json.loads(text).get("understanding", "")).strip() or row.situation
        )
    except Exception:  # noqa: BLE001 — пересказ best-effort, ситуация как фолбэк
        understanding = row.situation
    finally:
        await llm.aclose()

    row.answers = dict(body.raw_answers)
    row.understanding = understanding
    row.step = "understanding"
    await db.commit()
    return {"understanding": understanding}


class BriefBody(BaseModel):
    brief: dict[str, Any]


@router.put("/advisor/consultations/{consultation_id}/brief")
async def save_brief(
    consultation_id: str,
    body: BriefBody,
    db: AsyncSession = Depends(get_db),
    caller: str = Depends(get_user_key),
) -> dict[str, Any]:
    """Сохранить итоговый бриф (клиент присылает финал SSE /advisor как есть)."""
    row = await _require_own(db, consultation_id, caller)
    row.brief = body.brief
    row.step = "recommendation"
    await db.commit()
    return {"ok": True}


class OutcomeBody(BaseModel):
    outcome: str = Field(pattern="^(decided|need_info|postponed|rejected)$")


@router.post("/advisor/consultations/{consultation_id}/outcome")
async def set_outcome(
    consultation_id: str,
    body: OutcomeBody,
    db: AsyncSession = Depends(get_db),
    caller: str = Depends(get_user_key),
) -> dict[str, Any]:
    row = await _require_own(db, consultation_id, caller)
    row.outcome = body.outcome
    await db.commit()
    return {"ok": True}


@router.get("/advisor/consultations")
async def list_consultations(
    db: AsyncSession = Depends(get_db),
    caller: str = Depends(get_user_key),
) -> list[dict[str, Any]]:
    """Свои консультации, свежие сверху (для «Продолжить» и списка истории)."""
    rows = (
        await db.scalars(
            select(AdvisorConsultation)
            .where(AdvisorConsultation.owner == caller)
            .order_by(AdvisorConsultation.updated_at.desc())
            .limit(20)
        )
    ).all()
    return [
        {
            "id": r.id,
            "title": r.title,
            "step": r.step,
            "outcome": r.outcome,
            "updated_at": r.updated_at.isoformat() if r.updated_at else None,
        }
        for r in rows
    ]


@router.get("/advisor/consultations/{consultation_id}")
async def get_consultation(
    consultation_id: str,
    db: AsyncSession = Depends(get_db),
    caller: str = Depends(get_user_key),
) -> dict[str, Any]:
    """Полное состояние — фронт восстанавливает флоу с любого шага."""
    row = await _require_own(db, consultation_id, caller)
    return _serialize(row)


@router.delete("/advisor/consultations/{consultation_id}")
async def delete_consultation(
    consultation_id: str,
    db: AsyncSession = Depends(get_db),
    caller: str = Depends(get_user_key),
) -> dict[str, bool]:
    row = await _require_own(db, consultation_id, caller)
    await db.delete(row)
    await db.commit()
    return {"ok": True}
