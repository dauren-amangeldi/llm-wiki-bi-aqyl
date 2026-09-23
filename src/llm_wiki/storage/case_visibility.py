"""Case publication rules shared by API, status changes and startup backfill."""

from sqlalchemy import and_, cast, func, or_, select, text, update as sa_update
from sqlalchemy.dialects.postgresql import JSONB, array
from sqlalchemy.ext.asyncio import AsyncSession
from llm_wiki.storage.metadata import CaseRecord, ChunkEmbedding, FileRecord


def ready_case_clause():
    """Correlated EXISTS: only actual DONE sources make a case publishable."""
    # Expand this case's small source list, then look up files by primary key.
    # Scanning all DONE files for each case would make the workspace poll costly.
    sources = func.jsonb_array_elements_text(cast(CaseRecord.doc_ids, JSONB)).table_valued("value")
    return (
        select(FileRecord.file_id)
        .select_from(sources.join(FileRecord, FileRecord.file_id == sources.c.value))
        .where(FileRecord.status == "DONE")
        .correlate(CaseRecord)
        .exists()
    )


def public_case_clause():
    return and_(CaseRecord.sensitive.is_(False), ready_case_clause())


def visible_case_clause(caller: str):
    return or_(CaseRecord.owner == caller, public_case_clause())


async def has_ready_material(
    db: AsyncSession, doc_ids: list[str], caller: str | None = None
) -> bool:
    stmt = select(FileRecord.file_id).where(
        FileRecord.file_id.in_(doc_ids), FileRecord.status == "DONE"
    )
    if caller is not None:
        stmt = stmt.where(or_(FileRecord.sensitive.is_(False), FileRecord.owner == caller))
    return bool(await db.scalar(stmt.limit(1)))


async def case_visible(db: AsyncSession, case: CaseRecord, caller: str) -> bool:
    return case.owner == caller or (
        not case.sensitive and await has_ready_material(db, case.doc_ids or [])
    )


async def cascade_case_visibility(
    db: AsyncSession,
    doc_ids: list[str],
    *,
    sensitive: bool,
    owner: str | None,
    exclude_case_id: str | None = None,
) -> None:
    """Propagate a case's visibility to every material that belongs to it.

    Every time a case is saved we re-assert visibility over its files, their
    embedding chunks (what shared Q&A retrieval filters on) and the wiki
    page(s) each file created *on its own* (``created_pages``; shared
    ``updated_pages`` are left untouched).

    Мульти-кейс правило (QA-решение): материал может состоять в НЕСКОЛЬКИХ
    кейсах, поэтому кейс больше не единоличный источник правды. Файл приватен,
    только если ВСЕ содержащие его кейсы приватны: публикация любого кейса
    открывает материал; «сделать приватным» не прячет файл, пока тот входит в
    другой общий кейс. Public content has ``owner = NULL``; private content is
    owned by the case owner. The caller commits all visibility changes together.

    Args:
        exclude_case_id: id сохраняемого кейса — его СТАРАЯ строка в БД не
            должна голосовать за видимость (новое значение передано в
            ``sensitive``).
    """
    if not doc_ids:
        return

    # Файлы, у которых есть ДРУГОЙ публичный кейс — они остаются публичными,
    # даже когда сохраняемый кейс приватен.
    public_elsewhere: set[str] = set()
    if sensitive:
        stmt = select(CaseRecord.doc_ids).where(
            public_case_clause(),
            cast(CaseRecord.doc_ids, JSONB).op("?|")(array(doc_ids)),
        )
        if exclude_case_id:
            stmt = stmt.where(CaseRecord.id != exclude_case_id)
        doc_set = set(doc_ids)
        for (ids,) in (await db.execute(stmt)).all():
            public_elsewhere.update(d for d in (ids or []) if d in doc_set)

    private_docs = [d for d in doc_ids if sensitive and d not in public_elsewhere]
    public_docs = [d for d in doc_ids if d not in set(private_docs)]

    for batch, batch_sensitive in ((private_docs, True), (public_docs, False)):
        if not batch:
            continue
        new_owner = owner if batch_sensitive else None
        await db.execute(
            sa_update(FileRecord)
            .where(FileRecord.file_id.in_(batch))
            .values(sensitive=batch_sensitive, owner=new_owner)
        )
        await db.execute(
            sa_update(ChunkEmbedding)
            .where(ChunkEmbedding.file_id.in_(batch))
            .values(sensitive=batch_sensitive, owner=new_owner)
        )
        rows = (
            await db.execute(select(FileRecord.created_pages).where(FileRecord.file_id.in_(batch)))
        ).all()
        slugs = sorted({s for (pages,) in rows for s in (pages or [])})
        if slugs:
            await db.execute(
                text(
                    "UPDATE wiki_fts SET sensitive = :sensitive, owner = :owner WHERE slug = ANY(:slugs)"
                ),
                {"sensitive": batch_sensitive, "owner": new_owner, "slugs": slugs},
            )


async def privatize_unready_cases(db: AsyncSession, case_ids: list[str] | None = None) -> int:
    """One-way demotion, never automatically republish after a file finishes.

    Caller commits, so case/file/vector/wiki visibility changes are atomic.
    Old ownerless cases stay stored but hidden until ownership is restored.
    """
    stmt = select(CaseRecord).where(CaseRecord.sensitive.is_(False), ~ready_case_clause())
    if case_ids is not None:
        stmt = stmt.where(CaseRecord.id.in_(case_ids))
    rows = list(await db.scalars(stmt.order_by(CaseRecord.id).with_for_update()))
    for row in rows:
        row.sensitive = True
    await db.flush()
    for row in rows:
        await cascade_case_visibility(
            db, row.doc_ids or [], sensitive=True, owner=row.owner, exclude_case_id=row.id
        )
    return len(rows)
