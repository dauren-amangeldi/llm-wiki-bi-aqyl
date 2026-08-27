"""Лента уведомлений (Б1) — данные для колокольчика в шапке.

- ``GET  /notifications``       → {items, live, unread} (все вкладки разом —
  панель опрашивает одним запросом; фильтрация по вкладкам на клиенте)
- ``POST /notifications/read``  {ids?|all?} → {marked}

«Живые» строки (в работе) НЕ хранятся: они derive-ятся здесь из статусов
``files`` (активные стадии пайплайна) и ``artifacts`` (pending) на момент
запроса. Источник правды один — «вечно генерируется» в ленте невозможно:
пропала строка из files/artifacts → пропала из панели.
"""

from __future__ import annotations

from pydantic import BaseModel
from fastapi import Depends
from sqlalchemy import cast, or_, select
from sqlalchemy.dialects.postgresql import JSONB, array
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.api.deps import get_db, get_user_key
from llm_wiki.api.v1 import router
from llm_wiki.storage import notifications as notif_store
from llm_wiki.storage.metadata import ArtifactRecord, CaseRecord, FileRecord

# Активные стадии пайплайна (тот же набор, что у janitor-свипа).
_ACTIVE_FILE_STATUSES = [
    "RECEIVED", "STORED", "SEARCHED", "WRITTEN", "LINTED", "LOGGED",
    "PROCESSING", "PENDING",
    "received", "stored", "searched", "written", "linted", "logged",
    "processing", "pending",
]

# Дизайн: бэкендные статусы схлопнуты в 4 пользовательских шага («шаг 2/4»).
# stage — ключ для локализации подписи на фронте.
_FILE_STAGE: dict[str, tuple[int, str]] = {
    "RECEIVED": (1, "upload"),
    "PENDING": (1, "upload"),
    "PROCESSING": (1, "upload"),
    "STORED": (1, "upload"),
    "SEARCHED": (2, "analyze"),
    "WRITTEN": (3, "write"),
    "LINTED": (4, "finalize"),
    "LOGGED": (4, "finalize"),
}
_TOTAL_STEPS = 4


def _serialize_event(rec, read: bool) -> dict[str, object]:  # noqa: ANN001
    return {
        "id": rec.id,
        "section": rec.section,
        "family": rec.family,
        "event": rec.event,
        "entity_id": rec.entity_id,
        "title": rec.title,
        "detail": rec.detail,
        "actor": rec.actor,
        "meta": rec.meta or {},
        "created_at": rec.created_at.isoformat() if rec.created_at else None,
        "read": read,
    }


async def _live_rows(db: AsyncSession, caller: str) -> list[dict[str, object]]:
    """Derived «в работе»: файлы в активных стадиях + pending-артефакты +
    сводные строки по кейсам, чьи материалы ещё обрабатываются."""
    live: list[dict[str, object]] = []

    files = (
        await db.scalars(
            select(FileRecord)
            .where(
                FileRecord.status.in_(_ACTIVE_FILE_STATUSES),
                or_(FileRecord.owner.is_(None), FileRecord.owner == caller),
            )
            .order_by(FileRecord.created_at.desc())
            .limit(20)
        )
    ).all()

    # Кейсы, в которые входят обрабатываемые файлы → сводный прогресс кейса.
    cases: list[CaseRecord] = []
    if files:
        file_ids = [f.file_id for f in files]
        cases = (
            await db.scalars(
                select(CaseRecord).where(
                    cast(CaseRecord.doc_ids, JSONB).op("?|")(array(file_ids))
                )
            )
        ).all()
    case_by_file: dict[str, CaseRecord] = {}
    if cases:
        # Терминальные/активные статусы ВСЕХ материалов этих кейсов — одним
        # запросом (прогресс: N из M готово).
        all_doc_ids = sorted({d for c in cases for d in (c.doc_ids or [])})
        status_rows = (
            await db.execute(
                select(FileRecord.file_id, FileRecord.status).where(
                    FileRecord.file_id.in_(all_doc_ids)
                )
            )
        ).all()
        status_by_id = dict(status_rows)
        for case in cases:
            doc_ids = case.doc_ids or []
            done = sum(1 for d in doc_ids if status_by_id.get(d, "").upper() == "DONE")
            live.append(
                {
                    "section": "cases",
                    "entity_id": case.id,
                    "title": case.title,
                    "state": "running",
                    "done": done,
                    "total": len(doc_ids),
                }
            )
            for d in doc_ids:
                case_by_file[d] = case

    for f in files:
        step, stage = _FILE_STAGE.get(f.status.upper(), (1, "upload"))
        case = case_by_file.get(f.file_id)
        live.append(
            {
                "section": "materials",
                "entity_id": f.file_id,
                "title": f.display_name or f.original_name,
                "state": "running",
                "step": step,
                "total_steps": _TOTAL_STEPS,
                "stage": stage,
                "case_id": case.id if case else None,
            }
        )

    artifacts = (
        await db.scalars(
            select(ArtifactRecord)
            .where(
                ArtifactRecord.status == "pending",
                or_(
                    ArtifactRecord.requested_by.is_(None),
                    ArtifactRecord.requested_by == caller,
                ),
            )
            .order_by(ArtifactRecord.created_at.desc())
            .limit(20)
        )
    ).all()
    for art in artifacts:
        title = await notif_store.document_title(db, art.document_id)
        live.append(
            {
                "section": "artifacts",
                "entity_id": art.artifact_id,
                "title": title,
                # started_at ставит воркер — до него артефакт «в очереди».
                "state": "running" if art.started_at else "queued",
                "kind": art.kind,
                "document_id": art.document_id,
            }
        )

    return live


@router.get("/notifications")
async def get_notifications(
    db: AsyncSession = Depends(get_db),
    caller: str = Depends(get_user_key),
) -> dict[str, object]:
    """Вся лента одним запросом: события + живые генерации + непрочитанное."""
    events = await notif_store.list_events(db, caller, limit=60)
    return {
        "items": [_serialize_event(rec, read) for rec, read in events],
        "live": await _live_rows(db, caller),
        "unread": await notif_store.unread_counts(db, caller),
    }


class ReadBody(BaseModel):
    """Тело POST /notifications/read: конкретные ids или всё разом."""

    ids: list[int] = []
    all: bool = False


@router.post("/notifications/read")
async def mark_notifications_read(
    body: ReadBody,
    db: AsyncSession = Depends(get_db),
    caller: str = Depends(get_user_key),
) -> dict[str, int]:
    """Пометить прочитанным — «Прочитать все» или клик по строке."""
    marked = await notif_store.mark_read(
        db, caller, ids=body.ids or None, mark_all=body.all
    )
    return {"marked": marked}
