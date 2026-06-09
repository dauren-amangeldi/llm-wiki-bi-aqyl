"""Phase 1 + 2 endpoints — materials (documents) list, detail, status, and source management.

Maps FileRecord rows from SQLite to the Material schema consumed by the
llm-wiki-frontend (BI AQYL UI).  Authentication is bypassed at this phase;
the get_current_user dependency only provides the user dict for role checks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select, update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession

import numpy as np
from llm_wiki.api.deps import get_db
from llm_wiki.api.v1.deps import get_current_user
from llm_wiki.api.v1.schemas import (
    Dossier,
    Material,
    MaterialSource,
    SourcesResponse,
)
from llm_wiki.config import settings
from llm_wiki.llm.chunk_store import ChunkStore
from llm_wiki.llm.client import LLMClient
from llm_wiki.storage.metadata import FileRecord, get_file_record
from llm_wiki.utils.slugify import to_slug

logger = structlog.get_logger(__name__)

router = APIRouter()

_EXT_TO_CONTENT_TYPE: dict[str, Literal["pdf", "markdown", "video", "audio"]] = {
    ".pdf": "pdf",
    ".md": "markdown",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ext(original_name: str) -> str:
    return Path(original_name).suffix.lower()


def _fr_to_material(fr: FileRecord, snippet: str | None = None) -> Material:
    ext = _ext(fr.original_name)
    title = Path(fr.original_name).stem
    content_type = _EXT_TO_CONTENT_TYPE.get(ext, "markdown")
    return Material(
        document_id=fr.file_id,
        title=title,
        content_type=content_type,
        scope="internal",
        business_unit="HQ",
        status=fr.status,
        created_at=fr.created_at.isoformat(),
        updated_at=fr.updated_at.isoformat() if fr.updated_at else None,
        source_language="ru",
        tags=[],
        topic_ids=[],
        title_i18n={"ru": title},
        snippet=snippet,
        author=None,
        language="ru",
        classification=None,
    )


def _wiki_text(title: str) -> str | None:
    """Return the full text of the wiki page for *title*, or None if absent."""
    slug = to_slug(title)
    path = settings.wiki_dir / f"{slug}.md"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


async def _get_or_404(session: AsyncSession, document_id: str) -> FileRecord:
    record = await get_file_record(session, document_id)
    if record is None or record.status == "deleted":
        raise HTTPException(status_code=404, detail=f"Document {document_id!r} not found.")
    return record


# ---------------------------------------------------------------------------
# GET /documents  — list all materials
# ---------------------------------------------------------------------------


@router.get("/documents", response_model=list[Material], tags=["materials"])
async def list_documents(
    scope: Literal["all", "internal", "external"] = "all",
    q: str | None = None,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    language: str | None = None,
    bookmarks_only: bool = False,
    session: AsyncSession = Depends(get_db),
) -> list[Material]:
    """List all non-deleted FileRecords mapped to the Material schema.

    Query params mirror the frontend store filters so future phases can push
    filtering server-side.  For now *q* (substring match on title) is the only
    server-side filter; scope and bookmarks_only are handled client-side.
    """
    stmt = (
        select(FileRecord)
        .where(FileRecord.status != "deleted")
        .order_by(FileRecord.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    result = await session.execute(stmt)
    records = result.scalars().all()

    materials = [_fr_to_material(fr) for fr in records]

    if q:
        q_lower = q.lower()
        materials = [m for m in materials if q_lower in m.title.lower()]

    return materials


# ---------------------------------------------------------------------------
# GET /documents/{document_id}  — single material with snippet
# ---------------------------------------------------------------------------


@router.get("/documents/{document_id}", response_model=Material, tags=["materials"])
async def get_document(
    document_id: str,
    language: str | None = None,
    session: AsyncSession = Depends(get_db),
) -> Material:
    """Return a single Material; includes a 200-char wiki snippet when available."""
    fr = await _get_or_404(session, document_id)
    title = Path(fr.original_name).stem
    wiki_text = _wiki_text(title)
    snippet = wiki_text[:200] if wiki_text else None
    return _fr_to_material(fr, snippet=snippet)


# ---------------------------------------------------------------------------
# DELETE /documents/{document_id}  — soft delete (admin only)
# ---------------------------------------------------------------------------


@router.delete("/documents/{document_id}", status_code=200, tags=["materials"])
async def delete_document(
    document_id: str,
    user: Annotated[dict[str, Any], Depends(get_current_user)],
    session: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    """Soft-delete a document by setting status='deleted'.

    Only users with role 'admin' may delete documents.  Files on disk are
    left untouched — recovery is possible by updating the DB row directly.
    """
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin role required to delete documents.")

    fr = await _get_or_404(session, document_id)

    await session.execute(
        sa_update(FileRecord)
        .where(FileRecord.file_id == fr.file_id)
        .values(status="deleted")
    )
    await session.commit()

    logger.info("document_deleted", document_id=document_id, by=user.get("email"))
    return {"document_id": document_id, "status": "deleted"}


# ---------------------------------------------------------------------------
# GET /documents/{document_id}/sources
# ---------------------------------------------------------------------------


@router.get("/documents/{document_id}/sources", response_model=SourcesResponse, tags=["materials"])
async def get_sources(
    document_id: str,
    language: str | None = None,
    session: AsyncSession = Depends(get_db),
) -> SourcesResponse:
    """Return the list of source files for a document.

    Phase 1: one document = one source.  Phase 2 adds multi-file topics.
    """
    fr = await _get_or_404(session, document_id)
    ext = _ext(fr.original_name)
    content_type = _EXT_TO_CONTENT_TYPE.get(ext, "markdown")
    source = MaterialSource(
        title=fr.original_name,
        content_type=content_type,
        path=f"/api/v1/files/{fr.file_id}/raw",
        document_id=fr.file_id,
        status=fr.status,
    )
    return SourcesResponse(items=[source])


# ---------------------------------------------------------------------------
# GET /documents/{document_id}/dossier
# ---------------------------------------------------------------------------


@router.get("/documents/{document_id}/dossier", response_model=Dossier, tags=["materials"])
async def get_dossier(
    document_id: str,
    language: str | None = None,
    session: AsyncSession = Depends(get_db),
) -> Dossier:
    """Return a Dossier for the document.

    summary = first 500 chars of the synthesised wiki page (or None).
    page_count left as None until Phase 2 stores it during ingestion.
    """
    fr = await _get_or_404(session, document_id)
    title = Path(fr.original_name).stem
    wiki_text = _wiki_text(title)
    summary = wiki_text[:500] if wiki_text else None
    return Dossier(
        summary=summary,
        page_count=None,
        language="ru",
        status=fr.status,
    )


# ---------------------------------------------------------------------------
# GET /documents/{document_id}/related  — stub (Phase 4)
# ---------------------------------------------------------------------------


@router.get("/documents/{document_id}/related", tags=["materials"])
async def get_related(
    document_id: str,
    language: str | None = None,
    session: AsyncSession = Depends(get_db),
) -> dict[str, list[Any]]:
    """Return related materials based on cosine similarity of chunk embeddings.

    Algorithm:
    1. Get wiki slugs produced by this document's ingestion pipeline.
    2. Fetch stored embeddings for those chunks from ChromaDB.
    3. Average them into a single document-level embedding.
    4. Query ChromaDB for the 20 nearest chunks (global, including all docs).
    5. Group by slug, map slugs → FileRecords, exclude self, return top-5.
    """
    fr = await _get_or_404(session, document_id)
    wiki_slugs = list(set((fr.created_pages or []) + (fr.updated_pages or [])))

    if not wiki_slugs:
        return {"items": []}

    # Retrieve stored chunk embeddings for this document
    llm = LLMClient()
    try:
        chunk_store = ChunkStore(chroma_path=settings.chroma_dir, llm_client=llm)
        embeddings = chunk_store.get_embeddings_for_slugs(wiki_slugs)

        if not embeddings:
            return {"items": []}

        # Compute mean document embedding
        mean_emb: list[float] = np.mean(np.array(embeddings), axis=0).tolist()

        # Query ChromaDB for the 20 nearest chunks globally
        raw = chunk_store._col.query(  # type: ignore[attr-defined]
            query_embeddings=[mean_emb],
            n_results=min(20, chunk_store.count()),
            include=["metadatas"],  # type: ignore[list-item]
        )
    finally:
        await llm.aclose()

    # Collect related slugs (excluding the document's own slugs)
    own_slugs = set(wiki_slugs)
    related_slugs: list[str] = []
    seen_related: set[str] = set()
    for meta in (raw.get("metadatas") or [[]])[0]:
        slug = str(meta.get("slug", ""))
        if slug and slug not in own_slugs and slug not in seen_related:
            seen_related.add(slug)
            related_slugs.append(slug)
        if len(related_slugs) >= 20:
            break

    if not related_slugs:
        return {"items": []}

    # Map related slugs → FileRecords (scan all non-deleted records)
    result = await session.execute(
        select(FileRecord)
        .where(FileRecord.status != "deleted")
        .order_by(FileRecord.created_at.desc())
        .limit(500)
    )
    all_records = list(result.scalars().all())

    slug_set = set(related_slugs)
    matched: list[FileRecord] = []
    seen_fids: set[str] = {document_id}

    for candidate in all_records:
        if candidate.file_id in seen_fids:
            continue
        candidate_slugs = set(
            (candidate.created_pages or []) + (candidate.updated_pages or [])
        )
        if candidate_slugs & slug_set:
            seen_fids.add(candidate.file_id)
            matched.append(candidate)
        if len(matched) >= 5:
            break

    materials = [_fr_to_material(r) for r in matched]
    return {"items": [m.model_dump() for m in materials]}


# ---------------------------------------------------------------------------
# GET /documents/{document_id}/tags  — stub (Phase 6)
# ---------------------------------------------------------------------------


@router.get("/documents/{document_id}/tags", tags=["materials"])
async def get_document_tags(
    document_id: str,
    language: str | None = None,
    session: AsyncSession = Depends(get_db),
) -> dict[str, list[Any]]:
    """Return tags for a document.  Phase 1: empty stub (implemented in Phase 6)."""
    await _get_or_404(session, document_id)
    return {"tags": [], "suggestions": []}


# ===========================================================================
# Phase 2 — Upload status polling + source management
# ===========================================================================


class DocumentStatusResponse(BaseModel):
    status: str           # queued | processing | done | error
    progress: int         # 0-100
    error: str | None = None
    updated_at: str | None = None


# Rough progress mapping — good enough for a polling UI.
_INGESTION_PROGRESS: dict[str, tuple[str, int]] = {
    "RECEIVED": ("queued",     0),
    "STORED":   ("processing", 20),
    "SEARCHED": ("processing", 50),
    "WRITTEN":  ("processing", 80),
    "DONE":     ("done",      100),
    "FAILED":   ("error",      0),
}


# ---------------------------------------------------------------------------
# GET /documents/{document_id}/status
# ---------------------------------------------------------------------------


@router.get(
    "/documents/{document_id}/status",
    response_model=DocumentStatusResponse,
    tags=["materials"],
)
async def get_document_status(
    document_id: str,
    session: AsyncSession = Depends(get_db),
) -> DocumentStatusResponse:
    """Poll the ingestion status of a document.

    Returns a coarse ``progress`` (0-100) and a normalised ``status`` string
    that the frontend understands.  Poll every 2–3 seconds after upload.
    """
    fr = await _get_or_404(session, document_id)
    ui_status, progress = _INGESTION_PROGRESS.get(fr.status, ("processing", 50))
    error_msg = fr.status if ui_status == "error" else None
    return DocumentStatusResponse(
        status=ui_status,
        progress=progress,
        error=error_msg,
        updated_at=fr.updated_at.isoformat() if fr.updated_at else None,
    )


# ---------------------------------------------------------------------------
# DELETE /documents/{document_id}/sources/{source_id}
# ---------------------------------------------------------------------------


@router.delete(
    "/documents/{document_id}/sources/{source_id}",
    status_code=200,
    tags=["materials"],
)
async def delete_source(
    document_id: str,
    source_id: str,
    user: Annotated[dict[str, Any], Depends(get_current_user)],
    session: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    """Remove a source from a document.

    Phase 2: one document = one source, so ``source_id`` must equal
    ``document_id``.  Delegates to the document soft-delete.
    Phase 5+ will support multiple sources per document with independent
    deletion.
    """
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin role required to delete sources.")

    fr = await _get_or_404(session, document_id)

    if source_id != document_id:
        raise HTTPException(
            status_code=404,
            detail=f"Source {source_id!r} not found on document {document_id!r}.",
        )

    await session.execute(
        sa_update(FileRecord)
        .where(FileRecord.file_id == fr.file_id)
        .values(status="deleted")
    )
    await session.commit()

    logger.info("source_deleted", document_id=document_id, source_id=source_id, by=user.get("email"))
    return {"document_id": document_id, "source_id": source_id, "status": "deleted"}
