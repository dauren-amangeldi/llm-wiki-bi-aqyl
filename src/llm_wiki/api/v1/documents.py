"""Real document endpoints for the llm-wiki-frontend bridge (MVP)."""

from pathlib import Path
from typing import Literal

from fastapi import Depends, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.api.deps import get_db
from llm_wiki.api.v1 import router
from llm_wiki.config import settings
from llm_wiki.storage.metadata import FileRecord


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class Tag(BaseModel):
    """A tag that can be attached to a document."""

    id: str
    name: str


class Material(BaseModel):
    """Mirrors the Material interface in llm-wiki-frontend/src/stores/materials.ts."""

    document_id: str
    title: str
    content_type: Literal["pdf", "markdown"]
    scope: Literal["internal", "external"] = "internal"
    business_unit: str = "HQ"
    status: str
    created_at: str
    updated_at: str | None = None
    source_language: str = "ru"
    tags: list[Tag] = []
    topic_ids: list[str] = []
    title_i18n: dict[str, str] = {}
    snippet: str | None = None
    author: str | None = None
    language: str = "ru"
    classification: str | None = None
    possible_duplicate: bool = False


class Dossier(BaseModel):
    """Mirrors DossierData in llm-wiki-frontend/src/stores/modal.ts."""

    summary: str | None = None
    page_count: int | None = None
    language: str = "ru"
    status: str


# ---------------------------------------------------------------------------
# Mapper
# ---------------------------------------------------------------------------


def _file_record_to_material(fr: FileRecord) -> Material:
    """Convert a FileRecord ORM row into a Material response schema."""
    name = Path(fr.original_name).stem
    content_type: Literal["pdf", "markdown"] = (
        "pdf" if fr.original_name.lower().endswith(".pdf") else "markdown"
    )
    return Material(
        document_id=fr.file_id,
        title=name,
        content_type=content_type,
        scope="internal",
        business_unit="HQ",
        status=fr.status,
        created_at=fr.created_at.isoformat(),
        updated_at=fr.updated_at.isoformat() if fr.updated_at else None,
        source_language="ru",
        tags=[],
        topic_ids=[],
        title_i18n={"ru": name},
        snippet=None,
        author=None,
        language="ru",
        classification=None,
        possible_duplicate=False,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/documents", response_model=list[Material])
async def list_documents(
    q: str | None = None,
    language: str = "ru",
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
) -> list[Material]:
    """Return all non-rolled-back documents, optionally filtered by name."""
    stmt = select(FileRecord).where(FileRecord.status != "ROLLED_BACK")
    if q:
        stmt = stmt.where(FileRecord.original_name.ilike(f"%{q}%"))
    stmt = stmt.order_by(FileRecord.created_at.desc()).limit(limit).offset(offset)
    rows = (await db.execute(stmt)).scalars().all()
    return [_file_record_to_material(r) for r in rows]


@router.get("/documents/{document_id}", response_model=Material)
async def get_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
) -> Material:
    """Return a single document by ID."""
    fr = await db.get(FileRecord, document_id)
    if not fr:
        raise HTTPException(404, "Document not found")
    return _file_record_to_material(fr)


@router.get("/documents/{document_id}/dossier", response_model=Dossier)
async def get_dossier(
    document_id: str,
    language: str = "ru",
    db: AsyncSession = Depends(get_db),
) -> Dossier:
    """Return the full wiki-page content for a document (MVP: no truncation)."""
    fr = await db.get(FileRecord, document_id)
    if not fr:
        raise HTTPException(404, "Document not found")

    summary: str | None = None
    page_count: int | None = None

    if fr.created_pages:
        slug = fr.created_pages[0]
        wiki_path = settings.wiki_dir / f"{slug}.md"
        if wiki_path.exists():
            content = wiki_path.read_text(encoding="utf-8")
            summary = content
            page_count = max(1, len(content) // 3000)

    return Dossier(
        summary=summary,
        page_count=page_count,
        language="ru",
        status=fr.status,
    )


@router.post("/uploads", status_code=202)
async def uploads_alias(
    file: UploadFile,
    session: AsyncSession = Depends(get_db),
) -> object:
    """Alias for POST /api/v1/files — used by the frontend upload component."""
    from llm_wiki.api.routes import upload_file

    return await upload_file(file=file, session=session, _rate_check=None)
