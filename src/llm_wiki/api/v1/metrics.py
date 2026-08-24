"""Честные метрики для блока «Метрики» в настройках (BUG-11).

До этого фронт и бэк говорили на разных контрактах: MetricsSection ждал
``{stats, activity[], tags_distribution[]}``, мок отдавал 4 других ключа —
блок вечно показывал нули при десятках материалов в базе. Все данные уже
лежат в таблицах; здесь — только COUNT'ы и GROUP BY.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import Depends
from sqlalchemy import and_, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.api.deps import get_db, get_user_key
from llm_wiki.api.v1 import router
from llm_wiki.storage.metadata import (
    ArtifactRecord,
    CaseRecord,
    ChatRecord,
    FileRecord,
)


@router.get("/metrics")
async def metrics(
    db: AsyncSession = Depends(get_db),
    _caller: str = Depends(get_user_key),
) -> dict[str, Any]:
    total_docs = await db.scalar(select(func.count()).select_from(FileRecord)) or 0
    studied = (
        await db.scalar(
            select(func.count()).select_from(FileRecord).where(FileRecord.status == "DONE")
        )
        or 0
    )
    total_cases = await db.scalar(select(func.count()).select_from(CaseRecord)) or 0
    artifacts = (
        await db.scalar(
            select(func.count())
            .select_from(ArtifactRecord)
            .where(ArtifactRecord.status == "ready")
        )
        or 0
    )
    questions = (
        await db.scalar(
            select(func.count()).select_from(ChatRecord).where(ChatRecord.role == "user")
        )
        or 0
    )

    # Активность за 7 дней: вопросы в чате + запущенные генерации по дням.
    since = datetime.now(timezone.utc) - timedelta(days=6)
    q_rows = (
        await db.execute(
            select(
                func.date_trunc("day", ChatRecord.created_at).label("day"),
                func.count().label("n"),
            )
            .where(and_(ChatRecord.role == "user", ChatRecord.created_at >= since))
            .group_by(text("day"))
        )
    ).all()
    g_rows = (
        await db.execute(
            select(
                func.date_trunc("day", ArtifactRecord.created_at).label("day"),
                func.count().label("n"),
            )
            .where(ArtifactRecord.created_at >= since)
            .group_by(text("day"))
        )
    ).all()
    q_by_day = {r.day.date().isoformat(): int(r.n) for r in q_rows}
    g_by_day = {r.day.date().isoformat(): int(r.n) for r in g_rows}
    activity = []
    for i in range(7):
        day = (since + timedelta(days=i)).date().isoformat()
        activity.append(
            {
                "date": day,
                "questions": q_by_day.get(day, 0),
                "generations": g_by_day.get(day, 0),
            }
        )

    # Распределение тем: разворачиваем JSON-массив тегов кейсов на строки.
    tag_rows = (
        await db.execute(
            text(
                "SELECT tag.value AS name, count(*) AS n FROM cases c,"
                " json_array_elements_text(c.tags::json) AS tag"
                " GROUP BY tag.value ORDER BY n DESC LIMIT 8"
            )
        )
    ).all()
    tags_distribution = [{"name": r.name, "count": int(r.n)} for r in tag_rows]

    return {
        "stats": {
            "total_docs": int(total_docs),
            "total_cases": int(total_cases),
            "studied_cases": int(studied),
            "artifacts_generated": int(artifacts),
            "questions_asked": int(questions),
        },
        "activity": activity,
        "tags_distribution": tags_distribution,
    }
