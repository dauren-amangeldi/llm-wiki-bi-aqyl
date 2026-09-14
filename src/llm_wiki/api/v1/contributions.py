"""Contribution ranking from shared cases; private activity is never exposed."""

from datetime import datetime, timezone
from typing import Literal
from zoneinfo import ZoneInfo

from fastapi import Depends
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.api.deps import get_db, get_user_key
from llm_wiki.api.v1 import router
from llm_wiki.storage.metadata import CaseRecord, User

RANKING_TIMEZONE = "Asia/Almaty"


def month_start(now: datetime) -> datetime:
    return now.astimezone(ZoneInfo(RANKING_TIMEZONE)).replace(
        day=1, hour=0, minute=0, second=0, microsecond=0,
    )


def public_name(name: str | None) -> tuple[str, str]:
    """Use a human name if available; never fall back to email or account ID."""
    if not name or "@" in name:
        return "", ""
    parts = name.strip().split()
    if not parts:
        return "", ""
    shortened = parts[0] + (f" {parts[-1][0]}." if len(parts) > 1 else "")
    initials = parts[0][0] + (parts[-1][0] if len(parts) > 1 else "")
    return shortened, initials.upper()


@router.get("/contributions")
async def get_contributions(
    period: Literal["month", "all"] = "month",
    db: AsyncSession = Depends(get_db),
    caller: str = Depends(get_user_key),
) -> dict[str, object]:
    eligible = (
        CaseRecord.sensitive.is_(False),
        CaseRecord.owner.is_not(None),
        CaseRecord.owner.not_in(("", "anon")),
    )
    counts_query = select(
        CaseRecord.owner.label("owner"), func.count().label("cases"),
    ).where(*eligible)
    if period == "month":
        counts_query = counts_query.where(
            CaseRecord.created_at >= month_start(datetime.now(timezone.utc)),
        )
    counts = counts_query.group_by(CaseRecord.owner).cte("contribution_counts")
    ranked = select(
        counts.c.owner, counts.c.cases,
        func.rank().over(order_by=counts.c.cases.desc()).label("rank"),
        func.row_number().over(
            order_by=(counts.c.cases.desc(), counts.c.owner),
        ).label("position"),
    ).cte("contribution_ranked")
    # Rank everyone in SQL, return at most ten rows plus the caller's own row.
    rows = (await db.execute(
        select(ranked, User.name).outerjoin(User, User.id == ranked.c.owner)
        .where(or_(ranked.c.position <= 10, ranked.c.owner == caller))
        .order_by(ranked.c.position),
    )).mappings().all()
    mine = next((row for row in rows if row["owner"] == caller), None)
    has_cases_ever = bool(await db.scalar(
        select(select(CaseRecord.id).where(*eligible, CaseRecord.owner == caller).exists()),
    ))
    top = []
    for row in rows:
        if row["position"] > 10:
            continue
        name, initials = public_name(row["name"])
        top.append({
            "rank": row["rank"], "name": name, "initials": initials,
            "cases": row["cases"], "isMe": row["owner"] == caller,
        })
    return {
        "myRank": mine["rank"] if mine else None,
        "myCases": mine["cases"] if mine else 0,
        # No historical snapshots: do not fabricate changes in position.
        "rankDelta": None, "top": top, "hasCasesEver": has_cases_ever,
        "timezone": RANKING_TIMEZONE,
    }
