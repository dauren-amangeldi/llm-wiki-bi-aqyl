"""Mock endpoints for all remaining frontend routes (MVP stub layer)."""

from pathlib import Path

from fastapi import Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.api.deps import get_db
from llm_wiki.api.v1 import router
from llm_wiki.storage.metadata import FileRecord


# ── GET — empty / minimal responses ─────────────────────────────────────────


@router.get("/tags")
async def list_tags() -> list:
    """MOCK."""
    return []


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


@router.get("/artifacts")
async def artifacts(document_id: str | None = None) -> list:
    """MOCK."""
    return []


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



@router.get("/files/{file_id}/raw")
async def file_raw(
    file_id: str,
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Serve the original PDF/MD from the object store."""
    from llm_wiki.storage.object_store import get_object_store, raw_key

    fr = await db.get(FileRecord, file_id)
    if not fr:
        raise HTTPException(404, "File not found")
    ext = Path(fr.original_name).suffix
    store = get_object_store()
    key = raw_key(file_id, ext)
    if not store.exists(key):
        raise HTTPException(404, "Raw file not found in storage")
    media_type = "application/pdf" if ext.lower() == ".pdf" else "text/markdown"
    return Response(
        content=store.get_bytes(key),
        media_type=media_type,
        headers={"Content-Disposition": f'inline; filename="{fr.original_name}"'},
    )


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


@router.post("/studio/generate")
async def studio_generate(body: dict) -> dict:
    """MOCK: returns an empty presentation artifact."""
    return {
        "artifact_id": "mock-not-implemented",
        "kind": "presentation",
        "payload": {"slides": []},
    }


@router.post("/cards/generate")
async def cards_generate(body: dict) -> dict:
    """MOCK."""
    return {"artifact_id": "mock-not-implemented", "kind": "card", "payload": {}}


@router.post("/speech/podcast")
async def podcast(body: dict) -> dict:
    """MOCK."""
    return {"ok": True, "url": None}


@router.post("/images/generate")
async def images(body: dict) -> dict:
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


@router.patch("/documents/{document_id}")
async def update_doc(document_id: str, body: dict) -> dict:
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


@router.delete("/topics/{topic_id}/documents/{document_id}")
async def topic_remove(topic_id: str, document_id: str) -> dict:
    """MOCK."""
    return {"ok": True}
