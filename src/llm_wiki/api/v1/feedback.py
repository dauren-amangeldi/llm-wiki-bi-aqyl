"""Фидбэк-петля: 👍/👎 и «Сообщить об ошибке» (BUG-08/09).

Фронтовый components/FeedbackButtons.tsx уже давно шлёт ровно этот контракт
(entity_type, entity_id, vote) — запросы уходили в 404 и молча глотались.
"""

from __future__ import annotations

from typing import Any, Literal

import structlog
from fastapi import Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.api.deps import get_db, get_user_key
from llm_wiki.api.v1 import router
from llm_wiki.storage.metadata import FeedbackRecord

logger = structlog.get_logger(__name__)


class FeedbackBody(BaseModel):
    entity_type: str = Field(min_length=1, max_length=50)
    entity_id: str = Field(min_length=1, max_length=200)
    vote: Literal["up", "down"] | None = None
    reason: str | None = Field(default=None, max_length=200)
    comment: str | None = Field(default=None, max_length=4000)


@router.post("/feedback", status_code=201)
async def submit_feedback(
    body: FeedbackBody,
    db: AsyncSession = Depends(get_db),
    caller: str = Depends(get_user_key),
) -> dict[str, Any]:
    row = FeedbackRecord(
        owner=caller,
        entity_type=body.entity_type,
        entity_id=body.entity_id,
        vote=body.vote,
        reason=body.reason,
        comment=body.comment,
    )
    db.add(row)
    await db.commit()
    logger.info(
        "feedback_received",
        entity_type=body.entity_type,
        entity_id=body.entity_id,
        vote=body.vote,
        reason=body.reason,
    )
    return {"ok": True, "id": row.id}
