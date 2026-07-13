"""Cases (topic containers) CRUD endpoints."""

from datetime import UTC, datetime
from typing import Literal

from fastapi import Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy import update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.api.deps import get_db
from llm_wiki.api.v1 import router
from llm_wiki.storage.metadata import CaseRecord, find_similar_cases, refresh_case_embedding

CaseSource = Literal["internal", "external"]


class CaseBody(BaseModel):
    """Request/response body for case CRUD."""

    id: str | None = None
    title: str
    doc_ids: list[str] = []
    sensitive: bool = True
    source: CaseSource = "internal"


@router.get("/cases")
async def list_cases(db: AsyncSession = Depends(get_db)) -> list[dict[str, object]]:
    """Return all cases ordered by creation time."""
    rows = (await db.scalars(select(CaseRecord).order_by(CaseRecord.created_at))).all()
    return [
        {
            "id": r.id,
            "title": r.title,
            "doc_ids": r.doc_ids or [],
            "sensitive": r.sensitive,
            "source": r.source,
        }
        for r in rows
    ]


@router.post("/cases", status_code=201)
async def create_case(body: CaseBody, db: AsyncSession = Depends(get_db)) -> dict[str, object]:
    """Create a new case container."""
    now = datetime.now(UTC)
    case = CaseRecord(
        id=body.id or f"case-{int(now.timestamp() * 1000):x}-1",
        title=body.title.strip() or "Без названия",
        doc_ids=body.doc_ids,
        sensitive=body.sensitive,
        source=body.source,
        created_at=now,
        updated_at=now,
    )
    db.add(case)
    await db.commit()
    await refresh_case_embedding(db, case.id)
    return {
        "id": case.id,
        "title": case.title,
        "doc_ids": case.doc_ids,
        "sensitive": case.sensitive,
        "source": case.source,
    }


@router.put("/cases/{case_id}")
async def update_case(
    case_id: str, body: CaseBody, db: AsyncSession = Depends(get_db)
) -> dict[str, bool]:
    """Update case title, document membership, privacy, and source."""
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
            source=body.source,
            updated_at=datetime.now(UTC),
        )
    )
    await db.commit()
    await refresh_case_embedding(db, case_id)
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


@router.get("/cases/{case_id}/similar")
async def similar_cases(
    case_id: str, limit: int = 5, db: AsyncSession = Depends(get_db)
) -> list[dict[str, object]]:
    """Return the most similar other cases by document-content embedding.

    Empty list if the case has no documents yet, or none of them have
    finished processing (chunk embeddings not created yet) — this is a
    normal, expected state, not an error.
    """
    matches = await find_similar_cases(db, case_id, limit=limit)
    return [{"id": mid, "title": title, "similarity_pct": pct} for mid, title, pct in matches]
