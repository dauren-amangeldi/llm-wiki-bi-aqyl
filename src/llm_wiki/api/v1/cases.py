"""Cases (topic containers) CRUD endpoints."""

from datetime import datetime, timezone
from typing import Literal

import structlog
from fastapi import Depends, HTTPException, Query, Response
from pydantic import BaseModel
from sqlalchemy import and_, func, or_, select, update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.api.deps import get_db, get_user_key
from llm_wiki.api.v1 import router
from llm_wiki.storage import wiki_store
from llm_wiki.storage.metadata import CaseRecord, ChunkEmbedding, FileRecord
from llm_wiki.taxonomy import CASE_TAGS, clean_tags

logger = structlog.get_logger(__name__)


def _assert_can_edit(row: CaseRecord, caller: str) -> None:
    """A case (public or private) may only be modified by its author.

    ANY mutation — rename, tags, doc membership, privacy flip, delete — is
    author-only. Enforced ALWAYS (demo mode included): in demo the caller comes
    from the X-User-Email header the frontend consistently sends, so ownership
    attribution works there too. Legacy rows with owner=NULL predate ownership
    and stay editable by anyone (prod data gets owners on creation — Б2).
    """
    if row.owner and row.owner != caller:
        raise HTTPException(status_code=403, detail="Only the case author can modify this case")


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
    # Источник опыта: внутренний опыт BI или мировой. Двигает фильтр на главной.
    scope: Literal["internal", "external"] = "internal"


def _dispatch_autotag(case_id: str) -> None:
    """Queue LLM auto-tagging for a case — best-effort so a broker hiccup can't
    fail the request. The task itself skips cases that already have tags, so it
    won't clobber manual edits."""
    try:
        from llm_wiki.orchestrator.tasks import autotag_case

        autotag_case.delay(case_id)
    except Exception:  # noqa: BLE001
        pass


async def _cascade_case_visibility(
    db: AsyncSession, doc_ids: list[str], *, sensitive: bool, owner: str | None
) -> None:
    """Propagate a case's visibility to every material that belongs to it.

    A case is the single source of truth for the privacy of its nested files,
    their wiki pages and their embedding chunks. Every time a case is saved we
    re-assert that membership so two long-standing bugs can't happen:

    * publishing a case that was created **private** left its files / wiki /
      chunks private, so they never entered the shared search;
    * a file uploaded into an already-**public** case stayed private.

    Both are fixed by flipping ``(sensitive, owner)`` on each doc's
    ``FileRecord``, its embedding chunks (what shared Q&A retrieval filters on)
    and the wiki page(s) the file created *on its own*. Pages a public file
    merged into (``updated_pages``) are shared across many files and are left
    untouched — only ``created_pages`` (e.g. the ``private-{file_id}`` page) are
    flipped. Public content has ``owner = NULL``; private content is owned by the
    case owner. The caller commits the async work; the wiki store commits itself.
    """
    if not doc_ids:
        return
    new_owner = owner if sensitive else None

    await db.execute(
        sa_update(FileRecord)
        .where(FileRecord.file_id.in_(doc_ids))
        .values(sensitive=sensitive, owner=new_owner)
    )
    await db.execute(
        sa_update(ChunkEmbedding)
        .where(ChunkEmbedding.file_id.in_(doc_ids))
        .values(sensitive=sensitive, owner=new_owner)
    )
    rows = (
        await db.execute(
            select(FileRecord.created_pages).where(FileRecord.file_id.in_(doc_ids))
        )
    ).all()
    slugs = sorted({s for (pages,) in rows for s in (pages or [])})
    wiki_store.set_pages_visibility(slugs, sensitive=sensitive, owner=new_owner)


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
        term = q.strip()
        # Substring match OR trigram similarity (typo-tolerant, e.g. «маркетнг»).
        conds.append(
            or_(
                CaseRecord.title.ilike(f"%{term}%"),
                # word_similarity matches the query against the best word/extent
                # of the title, so a short typo scores high against a long title.
                func.word_similarity(term.lower(), func.lower(CaseRecord.title)) > 0.3,
            )
        )
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
            "owner": r.owner,
            "scope": r.scope or "internal",
            "description": r.description or "",
            "created_at": r.created_at.isoformat() if r.created_at else None,
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
        scope=body.scope,
        # Always attribute the author — "anon" included (clients without the
        # X-User-Email header). owner=NULL would leave the case editable by
        # everyone forever (see _assert_can_edit).
        owner=owner,
        created_at=now,
        updated_at=now,
    )
    db.add(case)
    # If the case is created already holding docs, align their visibility too.
    await _cascade_case_visibility(
        db, case.doc_ids, sensitive=case.sensitive, owner=case.owner
    )
    await db.commit()
    _dispatch_autotag(case.id)
    return {
        "id": case.id,
        "title": case.title,
        "doc_ids": case.doc_ids,
        "sensitive": case.sensitive,
        "tags": case.tags,
        "owner": case.owner,
        "scope": case.scope,
        "description": case.description or "",
        "created_at": case.created_at.isoformat() if case.created_at else None,
    }


@router.put("/cases/{case_id}")
async def update_case(
    case_id: str,
    body: CaseBody,
    db: AsyncSession = Depends(get_db),
    caller: str = Depends(get_user_key),
) -> dict[str, bool]:
    """Update case title and document membership."""
    row = await db.get(CaseRecord, case_id)
    if not row:
        raise HTTPException(status_code=404, detail="Case not found")
    _assert_can_edit(row, caller)
    # Состав материалов изменился → LLM-описание устарело: сбрасываем, а
    # _dispatch_autotag ниже перегенерит его по новому составу.
    docs_changed = set(body.doc_ids) != set(row.doc_ids or [])
    await db.execute(
        sa_update(CaseRecord)
        .where(CaseRecord.id == case_id)
        .values(
            title=body.title.strip() or row.title,
            doc_ids=body.doc_ids,
            tags=clean_tags(body.tags),
            sensitive=body.sensitive,
            scope=body.scope,
            **({"description": ""} if docs_changed else {}),
            updated_at=datetime.now(timezone.utc),
        )
    )
    # The case is the source of truth for its materials' privacy: re-assert it
    # over the (possibly newly-added) doc set so publishing propagates and new
    # uploads inherit the case's status. owner stays as stored (unchanged here).
    await _cascade_case_visibility(
        db, body.doc_ids, sensitive=body.sensitive, owner=row.owner
    )
    await db.commit()
    _dispatch_autotag(case_id)
    return {"ok": True}


@router.delete("/cases/{case_id}")
async def delete_case(
    case_id: str,
    db: AsyncSession = Depends(get_db),
    caller: str = Depends(get_user_key),
) -> dict[str, bool]:
    """Delete a case AND everything it brought into the knowledge base (BUG-02).

    До этого удалялась одна строка кейса: материалы, вики-страницы и
    эмбеддинги продолжали жить и находиться поиском — для приватного кейса
    это утечка «удалённого» содержимого.

    Синхронно (одна транзакция — из поиска кейс исчезает атомарно с ответом):
      - осиротевшие файлы кейса (не входящие в doc_ids других кейсов),
        их вики-страницы (created_pages) из wiki_fts и их chunk_embeddings;
      - артефакты кейса и его осиротевших документов;
      - твин-сессии кейса с сообщениями;
      - история чата кейса;
      - сама строка кейса.
    Фоново (light-очередь): только S3-объекты — внешнее медленное хранилище
    не должно держать HTTP-ответ; на поиск оно не влияет.

    ``updated_pages`` не трогаем: файл лишь дополнял чужую страницу — она
    существовала до него и принадлежит другому материалу.
    """
    from sqlalchemy import delete as sa_delete, text

    from llm_wiki.storage.metadata import ArtifactRecord, ChatRecord, TwinMessage, TwinSession

    row = await db.get(CaseRecord, case_id)
    if not row:
        raise HTTPException(status_code=404, detail="Case not found")
    _assert_can_edit(row, caller)

    file_ids = [d for d in (row.doc_ids or []) if d]

    # Файл сиротеет, только если ни один ДРУГОЙ кейс его не содержит.
    orphaned: list[str] = []
    if file_ids:
        other_cases = (
            await db.scalars(select(CaseRecord).where(CaseRecord.id != case_id))
        ).all()
        used_elsewhere = {d for c in other_cases for d in (c.doc_ids or [])}
        orphaned = [f for f in file_ids if f not in used_elsewhere]

    slugs: list[str] = []
    raw_keys: list[str] = []
    for fid in orphaned:
        fr = await db.get(FileRecord, fid)
        if fr is None:
            continue
        slugs.extend(s for s in (fr.created_pages or []) if s)
        if fr.raw_key:
            raw_keys.append(fr.raw_key)
        elif fr.original_name and "." in fr.original_name:
            # Legacy-строки без raw_key лежали по пути raw/<file_id><ext>.
            from llm_wiki.storage.object_store import legacy_raw_key

            raw_keys.append(legacy_raw_key(fid, "." + fr.original_name.rsplit(".", 1)[1]))

    if slugs:
        await db.execute(
            text("DELETE FROM wiki_fts WHERE slug = ANY(:slugs)"), {"slugs": slugs}
        )
        await db.execute(sa_delete(ChunkEmbedding).where(ChunkEmbedding.slug.in_(slugs)))
    if orphaned:
        # Ремень к подтяжкам: чанки, чей slug не попал в created_pages
        # (страница перезаписана другим прогоном), находятся по file_id.
        await db.execute(
            sa_delete(ChunkEmbedding).where(ChunkEmbedding.file_id.in_(orphaned))
        )
        await db.execute(
            sa_delete(ArtifactRecord).where(ArtifactRecord.document_id.in_(orphaned))
        )
        await db.execute(sa_delete(FileRecord).where(FileRecord.file_id.in_(orphaned)))

    await db.execute(sa_delete(ArtifactRecord).where(ArtifactRecord.document_id == case_id))
    session_ids = (
        await db.scalars(select(TwinSession.id).where(TwinSession.case_id == case_id))
    ).all()
    if session_ids:
        await db.execute(sa_delete(TwinMessage).where(TwinMessage.session_id.in_(session_ids)))
        await db.execute(sa_delete(TwinSession).where(TwinSession.id.in_(session_ids)))
    await db.execute(
        sa_delete(ChatRecord).where(
            and_(ChatRecord.scope_type == "case", ChatRecord.scope_id == case_id)
        )
    )
    await db.delete(row)
    await db.commit()

    if raw_keys:
        try:
            from llm_wiki.orchestrator.tasks import purge_case_objects

            purge_case_objects.delay(raw_keys)
        except Exception:  # noqa: BLE001 — брокер лёг: сироты в S3 доберёт
            # scripts/purge_orphans.py; из поиска содержимое уже исчезло.
            logger.warning("case_delete_s3_purge_not_queued", case_id=case_id)
    logger.info(
        "case_deleted_cascade",
        case_id=case_id,
        files_deleted=len(orphaned),
        pages_deleted=len(slugs),
        twin_sessions_deleted=len(session_ids),
    )
    return {"ok": True}


@router.delete("/cases/{case_id}/documents/{document_id}")
async def unlink_document(
    case_id: str,
    document_id: str,
    db: AsyncSession = Depends(get_db),
    caller: str = Depends(get_user_key),
) -> dict[str, bool]:
    """Remove a document (source) from a case.

    Unlinks the doc from the case's ``doc_ids`` and persists it — previously
    this was a mock, so a deleted source reappeared on reopening the case.
    The document itself is left in place; it just no longer belongs here.
    """
    row = await db.get(CaseRecord, case_id)
    if not row:
        raise HTTPException(status_code=404, detail="Case not found")
    _assert_can_edit(row, caller)
    doc_ids = [d for d in (row.doc_ids or []) if d != document_id]
    await db.execute(
        sa_update(CaseRecord)
        .where(CaseRecord.id == case_id)
        .values(doc_ids=doc_ids, updated_at=datetime.now(timezone.utc))
    )
    await db.commit()
    return {"ok": True}


@router.get("/tags")
async def list_tags() -> list[dict[str, str]]:
    """The fixed case-tag taxonomy — name + description, for the tag picker/filter."""
    return [{"name": name, "description": desc} for name, desc in CASE_TAGS]
