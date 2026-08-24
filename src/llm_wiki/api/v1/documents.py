"""Real document endpoints for the llm-wiki-frontend bridge (MVP)."""

from pathlib import Path
from typing import Literal

from fastapi import Depends, Form, HTTPException, Response, UploadFile
from pydantic import BaseModel
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.api.deps import get_db, get_user_key
from llm_wiki.api.v1 import router
from llm_wiki.storage import wiki_store
from llm_wiki.storage.metadata import FileRecord

_MOCK_TAG_SUGGESTIONS: list[dict[str, str]] = [
    {"id": "t-strategy", "name": "Стратегия"},
    {"id": "t-dev", "name": "Девелопмент"},
    {"id": "t-finance", "name": "Финансы"},
    {"id": "t-risk", "name": "Риски"},
    {"id": "t-quality", "name": "Качество"},
    {"id": "t-sales", "name": "Продажи"},
]


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
    content_type: Literal["pdf", "markdown", "docx", "text", "audio"]
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
    sensitive: bool = False


class Dossier(BaseModel):
    """Mirrors DossierData in llm-wiki-frontend/src/stores/modal.ts."""

    summary: str | None = None
    page_count: int | None = None
    language: str = "ru"
    status: str


class DocumentText(BaseModel):
    """Full processed text of a document (markdown from its wiki page)."""

    content: str | None = None
    format: str = "markdown"
    slug: str | None = None


# ---------------------------------------------------------------------------
# Mapper
# ---------------------------------------------------------------------------


_AUDIO_EXTS = (".mp3", ".ogg", ".wav", ".m4a", ".webm")


def _content_type_for(name: str) -> Literal["pdf", "markdown", "docx", "text", "audio"]:
    """Map an original filename to a Material content_type (drives the UI icon)."""
    n = (name or "").lower()
    if n.endswith(".pdf"):
        return "pdf"
    if n.endswith(".docx"):
        return "docx"
    if n.endswith(".txt"):
        return "text"
    if n.endswith(_AUDIO_EXTS):
        return "audio"
    return "markdown"


def _file_record_to_material(fr: FileRecord) -> Material:
    """Convert a FileRecord ORM row into a Material response schema."""
    # Автонейминг (BUG-обс. ревизии): человеческое название из LLM,
    # фолбэк — имя файла без расширения.
    name = fr.display_name or Path(fr.original_name).stem
    content_type = _content_type_for(fr.original_name)
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
        sensitive=fr.sensitive,
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
    caller: str = Depends(get_user_key),
) -> list[Material]:
    """Return non-rolled-back documents visible to the caller.

    Sensitive documents are listed only for their owner.
    """
    stmt = select(FileRecord).where(
        FileRecord.status != "ROLLED_BACK",
        or_(FileRecord.sensitive.is_(False), FileRecord.owner == caller),
    )
    if q and q.strip():
        term = q.strip()
        # Substring OR trigram similarity so typos still match (e.g. «маркетнг»).
        stmt = stmt.where(
            or_(
                FileRecord.original_name.ilike(f"%{term}%"),
                func.word_similarity(term.lower(), func.lower(FileRecord.original_name)) > 0.3,
            )
        )
    stmt = stmt.order_by(FileRecord.created_at.desc()).limit(limit).offset(offset)
    rows = (await db.execute(stmt)).scalars().all()
    return [_file_record_to_material(r) for r in rows]


@router.get("/documents/{document_id}", response_model=Material)
async def get_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    caller: str = Depends(get_user_key),
) -> Material:
    """Return a single document by ID (sensitive docs only for their owner)."""
    fr = await db.get(FileRecord, document_id)
    # 404 (not 403) for a non-owner so a private doc's existence isn't revealed.
    if not fr or (fr.sensitive and fr.owner != caller):
        raise HTTPException(404, "Document not found")
    return _file_record_to_material(fr)


@router.get("/documents/{document_id}/dossier", response_model=Dossier)
async def get_dossier(
    document_id: str,
    language: str = "ru",
    db: AsyncSession = Depends(get_db),
    caller: str = Depends(get_user_key),
) -> Dossier:
    """Return the full wiki-page content for a document (MVP: no truncation)."""
    fr = await db.get(FileRecord, document_id)
    if not fr:
        raise HTTPException(404, "Document not found")
    # A private document's content is readable only by its owner.
    if fr.sensitive and fr.owner != caller:
        raise HTTPException(404, "Document not found")

    summary: str | None = None
    page_count: int | None = None

    if fr.created_pages:
        from llm_wiki.storage import wiki_store

        slug = fr.created_pages[0]
        content = wiki_store.get_page(slug)
        if content is not None:
            summary = content
            page_count = max(1, len(content) // 3000)

    return Dossier(
        summary=summary,
        page_count=page_count,
        language="ru",
        status=fr.status,
    )


@router.get("/documents/{document_id}/text", response_model=DocumentText)
async def get_document_text(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    caller: str = Depends(get_user_key),
) -> DocumentText:
    """Return the full processed markdown text of a document."""
    fr = await db.get(FileRecord, document_id)
    if not fr:
        raise HTTPException(status_code=404, detail="Document not found")
    # A private document's content is readable only by its owner.
    if fr.sensitive and fr.owner != caller:
        raise HTTPException(status_code=404, detail="Document not found")

    all_pages = list(fr.created_pages or []) + list(fr.updated_pages or [])
    if not all_pages:
        return DocumentText(content=None, slug=None)

    from llm_wiki.storage import wiki_store

    slug = all_pages[0]
    content = wiki_store.get_page(slug)
    if content is None:
        return DocumentText(content=None, slug=slug)

    return DocumentText(content=content, format="markdown", slug=slug)


@router.get("/documents/{document_id}/tags")
async def get_document_tags(
    document_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, list[dict[str, str]]]:
    """Return per-document tags and global suggestions (MVP: tags empty)."""
    fr = await db.get(FileRecord, document_id)
    if not fr:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"tags": [], "suggestions": _MOCK_TAG_SUGGESTIONS[:4]}


@router.get("/documents/{document_id}/related")
async def get_related_documents(
    document_id: str,
    limit: int = 4,
    db: AsyncSession = Depends(get_db),
) -> dict[str, list[dict[str, object]]]:
    """Return documents with similar titles (word overlap)."""
    fr = await db.get(FileRecord, document_id)
    if not fr:
        return {"items": []}

    source_words = set(
        Path(fr.original_name).stem.lower().replace("_", " ").replace("-", " ").split()
    )
    if not source_words:
        return {"items": []}

    stmt = select(FileRecord).where(
        FileRecord.file_id != document_id,
        FileRecord.status.notin_(["FAILED", "ROLLED_BACK"]),
    )
    rows = (await db.scalars(stmt.limit(50))).all()

    scored: list[tuple[float, FileRecord]] = []
    for row in rows:
        words = set(
            Path(row.original_name).stem.lower().replace("_", " ").replace("-", " ").split()
        )
        overlap = len(source_words & words) / max(len(source_words | words), 1)
        scored.append((overlap, row))

    scored.sort(key=lambda item: item[0], reverse=True)

    items: list[dict[str, object]] = []
    for score, row in scored[:limit]:
        name = Path(row.original_name).stem
        items.append(
            {
                "document_id": row.file_id,
                "title": name,
                "content_type": _content_type_for(row.original_name),
                "score": round(score, 3),
            }
        )

    return {"items": items}


@router.post("/uploads", status_code=202)
async def uploads_alias(
    file: UploadFile,
    response: Response,
    sensitive: bool = Form(False),
    session: AsyncSession = Depends(get_db),
    owner: str = Depends(get_user_key),
) -> object:
    """Alias for POST /api/v1/files — used by the frontend upload component."""
    from llm_wiki.api.routes import upload_file

    return await upload_file(
        file=file,
        response=response,
        sensitive=sensitive,
        session=session,
        owner=owner,
        _rate_check=None,
    )


class DocumentPatch(BaseModel):
    """Editable document fields (currently just the display name)."""

    title: str | None = None


@router.patch("/documents/{document_id}")
async def rename_document(
    document_id: str,
    body: DocumentPatch,
    db: AsyncSession = Depends(get_db),
    caller: str = Depends(get_user_key),
) -> dict[str, bool]:
    """Rename a source (was a mock that dropped the new name).

    Updates the file's human ``original_name`` (extension preserved) so the
    sources list and answer citations show the new name, and syncs the title of
    the wiki page(s) the file created on its own (``created_pages``) so the
    reader header and citation footer follow. Pages a public file merged into
    (shared across files) are left untouched. A sensitive file can only be
    renamed by its owner.
    """
    fr = await db.get(FileRecord, document_id)
    if not fr or (fr.sensitive and fr.owner != caller):
        raise HTTPException(status_code=404, detail="Document not found")
    new_name = (body.title or "").strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="title must not be empty")

    # Переименование правит ТОЛЬКО отображаемое имя; original_name остаётся
    # честным именем файла (скачивание оригинала, расширение).
    fr.display_name = new_name
    await db.commit()
    # Retitle the file's own wiki page(s); merged/shared pages stay as they are.
    wiki_store.set_pages_title(list(fr.created_pages or []), new_name)
    return {"ok": True}
