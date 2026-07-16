"""Leaderboard endpoints — aggregate case activity per employee."""

from fastapi import Depends
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.api.deps import get_db
from llm_wiki.api.v1 import router
from llm_wiki.storage.metadata import CaseRecord, User


class LeaderboardEntry(BaseModel):
    """One row of the "top employees by case count" leaderboard."""

    user_id: str
    name: str
    case_count: int


@router.get("/leaderboard/cases", response_model=list[LeaderboardEntry])
async def leaderboard_cases(
    limit: int = 10, db: AsyncSession = Depends(get_db)
) -> list[LeaderboardEntry]:
    """Return employees ranked by number of cases they created.

    Cases with ``owner_id IS NULL`` (created before the column existed, or
    via a client that sent no identity header) are excluded — an
    unattributed case can't be credited to anyone.
    """
    stmt = (
        select(
            CaseRecord.owner_id,
            func.coalesce(User.name, CaseRecord.owner_id).label("name"),
            func.count(CaseRecord.id).label("case_count"),
        )
        .join(User, User.id == CaseRecord.owner_id, isouter=True)
        .where(CaseRecord.owner_id.is_not(None))
        .group_by(CaseRecord.owner_id, User.name)
        .order_by(func.count(CaseRecord.id).desc())
        .limit(limit)
    )
    rows = (await db.execute(stmt)).all()
    return [
        LeaderboardEntry(user_id=r.owner_id, name=r.name, case_count=r.case_count)
        for r in rows
    ]
