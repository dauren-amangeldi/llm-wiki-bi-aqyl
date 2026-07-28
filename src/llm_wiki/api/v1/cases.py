"""Cases (topic containers) CRUD endpoints."""

from datetime import datetime, timezone
from typing import Literal

from fastapi import Depends, HTTPException, Query, Response
from pydantic import BaseModel
from sqlalchemy import and_, func, or_, select, update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.api.deps import get_db, get_user_key
from llm_wiki.api.v1 import router
from llm_wiki.storage.metadata import CaseRecord
from llm_wiki.taxonomy import CASE_TAGS, clean_tags


class CaseBody(BaseModel):
    """Request/response body for case CRUD."""

    id: str | None = None
    title: str
    doc_ids: list[str] = []
    # Private case: only its owner can list/open it. The frontend sends this
    # explicitly (new cases default private there); the backend default is False
    # so a client that omits it gets a normal, visible case. File-level privacy
    # (owner-scoped chunks) is what actually protects sensitive content.
    sensitive: bool = False
    # Fixed-taxonomy tags; unknown tags are dropped server-side (see clean_tags).
    tags: list[str] = []


def _dispatch_autotag(case_id: str) -> None:
    """Queue LLM auto-tagging for a case — best-effort so a broker hiccup can't
    fail the request. The task itself skips cases that already have tags, so it
    won't clobber manual edits."""
    try:
        from llm_wiki.orchestrator.tasks import autotag_case

        autotag_case.delay(case_id)
    except Exception:  # noqa: BLE001
        pass


@router.get("/cases")
async def list_cases(
    response: Response,
    db: AsyncSession = Depends(get_db),
    caller: str = Depends(get_user_key),
    q: str | None = Query(
        None, description="Case-insensitive substring match on the case title"
    ),
    category: Literal["all", "private", "public"] = Query(
        "all", description="Все / Приватные (свои) / Общие"
    ),
    limit: int | None = Query(
        None, ge=1, le=200, description="Page size; omit to return all (legacy behaviour)"
    ),
    offset: int = Query(0, ge=0, description="Number of rows to skip (pagination)"),
) -> list[dict[str, object]]:
    """Return cases visible to the caller, with search + category filter + pagination.

    Visibility is always enforced: the caller sees public cases and their own
    private ones. ``category`` narrows within that — ``all`` (both), ``public``
    (shared only) or ``private`` (the caller's own). ``q`` is a case-insensitive
    substring match on the title. Newest first (``created_at`` desc). The total
    number of matches (ignoring ``limit``/``offset``) is returned in the
    ``X-Total-Count`` header so the client can render pagination.
    """
    conds = [or_(CaseRecord.sensitive.is_(False), CaseRecord.owner == caller)]
    if category == "public":
        conds.append(CaseRecord.sensitive.is_(False))
    elif category == "private":
        conds.append(and_(CaseRecord.sensitive.is_(True), CaseRecord.owner == caller))
    if q and q.strip():
        conds.append(CaseRecord.title.ilike(f"%{q.strip()}%"))
    where = and_(*conds)

    total = await db.scalar(select(func.count()).select_from(CaseRecord).where(where))
    response.headers["X-Total-Count"] = str(total or 0)

    stmt = select(CaseRecord).where(where).order_by(CaseRecord.created_at.desc())
    if limit is not None:
        stmt = stmt.offset(offset).limit(limit)
    rows = (await db.scalars(stmt)).all()
    return [
        {
            "id": r.id,
            "title": r.title,
            "doc_ids": r.doc_ids or [],
            "sensitive": r.sensitive,
            "tags": r.tags or [],
        }
        for r in rows
    ]


@router.post("/cases", status_code=201)
async def create_case(
    body: CaseBody,
    db: AsyncSession = Depends(get_db),
    owner: str = Depends(get_user_key),
) -> dict[str, object]:
    """Create a new case container."""
    now = datetime.now(timezone.utc)
    case = CaseRecord(
        id=body.id or f"case-{int(now.timestamp() * 1000):x}-1",
        title=body.title.strip() or "Без названия",
        doc_ids=body.doc_ids,
        tags=clean_tags(body.tags),
        sensitive=body.sensitive,
        owner=owner if owner != "anon" else None,
        created_at=now,
        updated_at=now,
    )
    db.add(case)
    await db.commit()
    _dispatch_autotag(case.id)
    return {
        "id": case.id,
        "title": case.title,
        "doc_ids": case.doc_ids,
        "sensitive": case.sensitive,
        "tags": case.tags,
    }


@router.put("/cases/{case_id}")
async def update_case(
    case_id: str, body: CaseBody, db: AsyncSession = Depends(get_db)
) -> dict[str, bool]:
    """Update case title and document membership."""
    row = await db.get(CaseRecord, case_id)
    if not row:
        raise HTTPException(status_code=404, detail="Case not found")
    await db.execute(
        sa_update(CaseRecord)
        .where(CaseRecord.id == case_id)
        .values(
            title=body.title.strip() or row.title,
            doc_ids=body.doc_ids,
            tags=clean_tags(body.tags),
            sensitive=body.sensitive,
            updated_at=datetime.now(timezone.utc),
        )
    )
    await db.commit()
    _dispatch_autotag(case_id)
    return {"ok": True}


@router.delete("/cases/{case_id}")
async def delete_case(case_id: str, db: AsyncSession = Depends(get_db)) -> dict[str, bool]:
    """Delete a case by id."""
    row = await db.get(CaseRecord, case_id)
    if not row:
        raise HTTPException(status_code=404, detail="Case not found")
    await db.delete(row)
    await db.commit()
    return {"ok": True}


@router.get("/tags")
async def list_tags() -> list[dict[str, str]]:
    """The fixed case-tag taxonomy — name + description, for the tag picker/filter."""
    return [{"name": name, "description": desc} for name, desc in CASE_TAGS]
