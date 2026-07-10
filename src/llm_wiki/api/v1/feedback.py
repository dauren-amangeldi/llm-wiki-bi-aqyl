"""Feedback endpoint — 👍/👎 votes on AI artefacts (R2-3)."""

from __future__ import annotations

from typing import Literal

from fastapi import Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.api.deps import get_db, get_user_key
from llm_wiki.api.v1 import router
from llm_wiki.storage.metadata import Feedback


class FeedbackRequest(BaseModel):
    entity_type: Literal["similar_case", "twin_verdict"]
    entity_id: str = Field(min_length=1, max_length=200)
    vote: Literal[1, -1]
    comment: str = Field(default="", max_length=1000)


@router.post("/feedback")
async def submit_feedback(
    body: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
    user_key: str = Depends(get_user_key),
) -> dict[str, bool]:
    db.add(
        Feedback(
            entity_type=body.entity_type,
            entity_id=body.entity_id,
            vote=body.vote,
            comment=body.comment,
            created_by=user_key,
        )
    )
    await db.commit()
    return {"ok": True}
