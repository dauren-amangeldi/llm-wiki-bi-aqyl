"""Mock endpoints for all remaining frontend routes (MVP stub layer)."""

from fastapi import Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.api.deps import get_db
from llm_wiki.api.v1 import router
from llm_wiki.storage.metadata import FileRecord


# ── GET — empty / minimal responses ─────────────────────────────────────────


# GET /tags now lives in api/v1/cases.py — returns the real taxonomy.


@router.get("/documents/{document_id}/sources")
async def doc_sources(
    document_id: str,
    db: AsyncSession = Depends(get_db),
) -> list:
    """Return one source entry pointing at the raw file (minimal implementation)."""
    fr = await db.get(FileRecord, document_id)
    if not fr:
        return []
    return [
        {
            "title": fr.original_name,
            "content_type": (
                "pdf" if fr.original_name.lower().endswith(".pdf") else "markdown"
            ),
            "path": f"/api/v1/files/{document_id}/raw",
            "document_id": document_id,
            "status": fr.status,
        }
    ]


@router.get("/notifications")
async def notifications() -> list:
    """MOCK."""
    return []


@router.get("/guidelines")
async def guidelines() -> dict:
    """MOCK."""
    return {"cards": []}


@router.get("/metrics")
async def metrics(db: AsyncSession = Depends(get_db)) -> dict:
    """Return real document count; all other counters are zero."""
    total = await db.scalar(select(func.count()).select_from(FileRecord))
    return {
        "total_documents": total or 0,
        "total_queries_today": 0,
        "total_artifacts": 0,
        "last_activity": None,
    }



# GET /files/{file_id}/raw now lives in api/routes.py — the real endpoint adds
# owner/sensitive access control + all file types. Removed from the mock layer.


# ── POST — 200 with stub payload ─────────────────────────────────────────────


@router.post("/chat")
async def dashboard_chat(body: dict) -> dict:
    """MOCK: global dashboard chat stub."""
    return {
        "answer": (
            "Глобальный чат пока недоступен. "
            "Откройте материал и задайте вопрос там."
        ),
        "sources": [],
    }


@router.post("/speech/podcast")
async def podcast(body: dict) -> dict:
    """MOCK."""
    return {"ok": True, "url": None}


@router.post("/studio/compare")
async def compare(body: dict) -> dict:
    """MOCK."""
    return {"ok": True, "diff": []}


@router.post("/notifications/{nid}/read")
async def notif_read(nid: str) -> dict:
    """MOCK."""
    return {"ok": True}


# ── PUT / PATCH / DELETE — 200 with {ok: true} ───────────────────────────────


@router.put("/tags/documents/{document_id}")
async def set_tags(document_id: str, body: dict) -> dict:
    """MOCK."""
    return {"ok": True}


@router.delete("/tags/documents/{document_id}/{tag_id}")
async def remove_tag(document_id: str, tag_id: str) -> dict:
    """MOCK."""
    return {"ok": True}


@router.delete("/documents/{document_id}")
async def delete_doc(
    document_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Soft-delete: set status to ROLLED_BACK (filtered from GET /documents)."""
    fr = await db.get(FileRecord, document_id)
    if fr:
        fr.status = "ROLLED_BACK"
        await db.commit()
    return {"ok": True}
