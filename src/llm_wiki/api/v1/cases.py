"""Cases (topic containers) CRUD endpoints."""

from datetime import datetime, timezone

from fastapi import Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select, update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.api.deps import get_db
from llm_wiki.api.v1 import router
from llm_wiki.storage.metadata import CaseRecord


class CaseBody(BaseModel):
    """Request/response body for case CRUD."""

    id: str | None = None
    title: str
    doc_ids: list[str] = []


@router.get("/cases")
async def list_cases(db: AsyncSession = Depends(get_db)) -> list[dict[str, object]]:
    """Return all cases ordered by creation time."""
    rows = (await db.scalars(select(CaseRecord).order_by(CaseRecord.created_at))).all()
    return [{"id": r.id, "title": r.title, "doc_ids": r.doc_ids or []} for r in rows]


@router.post("/cases", status_code=201)
async def create_case(body: CaseBody, db: AsyncSession = Depends(get_db)) -> dict[str, object]:
    """Create a new case container."""
    now = datetime.now(timezone.utc)
    case = CaseRecord(
        id=body.id or f"case-{int(now.timestamp() * 1000):x}-1",
        title=body.title.strip() or "Без названия",
        doc_ids=body.doc_ids,
        created_at=now,
        updated_at=now,
    )
    db.add(case)
    await db.commit()
    return {"id": case.id, "title": case.title, "doc_ids": case.doc_ids}


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
