"""Cases (topic containers) CRUD endpoints."""

from datetime import datetime, timezone

from fastapi import Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import or_, select, update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.api.deps import get_db, get_user_key
from llm_wiki.api.v1 import router
from llm_wiki.storage.metadata import CaseRecord


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


@router.get("/cases")
async def list_cases(
    db: AsyncSession = Depends(get_db),
    caller: str = Depends(get_user_key),
) -> list[dict[str, object]]:
    """Return cases visible to the caller — private cases only for their owner."""
    stmt = (
        select(CaseRecord)
        .where(or_(CaseRecord.sensitive.is_(False), CaseRecord.owner == caller))
        .order_by(CaseRecord.created_at)
    )
    rows = (await db.scalars(stmt)).all()
    return [
        {"id": r.id, "title": r.title, "doc_ids": r.doc_ids or [], "sensitive": r.sensitive}
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
        sensitive=body.sensitive,
        owner=owner if owner != "anon" else None,
        created_at=now,
        updated_at=now,
    )
    db.add(case)
    await db.commit()
    return {
        "id": case.id,
        "title": case.title,
        "doc_ids": case.doc_ids,
        "sensitive": case.sensitive,
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
            sensitive=body.sensitive,
            updated_at=datetime.now(timezone.utc),
        )
    )
    await db.commit()
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
