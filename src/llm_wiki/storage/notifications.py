"""Уведомления (Б1) — запись событий и выборка ленты для колокольчика.

Здесь только ПЕРСИСТЕНТНЫЕ события: терминальные исходы генераций
(done/failed) и социальные события (смена приватности кейса). «В работе»
никогда не пишется — живые строки derive-ятся из статусов files/artifacts в
``api/v1/notifications.py`` на момент запроса.

Все ``notify_*`` эмиттеры — best-effort: любой их сбой логируется и глотается,
уведомление не имеет права уронить пайплайн или API-запрос.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import structlog
from sqlalchemy import and_, delete as sa_delete, func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.config import settings
from llm_wiki.storage.metadata import (
    CaseRecord,
    FileRecord,
    NotificationRead,
    NotificationRecord,
    case_for_file,
)

logger = structlog.get_logger(__name__)

SECTIONS = ("cases", "materials", "artifacts")

# Терминальные статусы файла — кейс считается «обработанным», когда все его
# материалы дошли до одного из них (и хотя бы один — до DONE).
_FILE_TERMINAL = ("DONE", "FAILED", "ROLLED_BACK")


# ---------------------------------------------------------------------------
# Запись (in-place upsert)
# ---------------------------------------------------------------------------


async def upsert_event(
    session: AsyncSession,
    *,
    section: str,
    family: str,
    event: str,
    entity_id: str,
    title: str,
    recipient: str | None = None,
    actor: str | None = None,
    detail: str | None = None,
    meta: dict | None = None,
) -> None:
    """Одна строка на (section, entity_id, family): существующая обновляется.

    Смена события (failed→done, published→privated) сбрасывает отметки чтения
    — строка снова непрочитана — и поднимает её наверх ленты (created_at=now).
    """
    now = datetime.now(timezone.utc)
    row = (
        await session.execute(
            select(NotificationRecord).where(
                NotificationRecord.section == section,
                NotificationRecord.entity_id == entity_id,
                NotificationRecord.family == family,
            )
        )
    ).scalar_one_or_none()
    if row is None:
        session.add(
            NotificationRecord(
                section=section,
                family=family,
                event=event,
                entity_id=entity_id,
                title=title,
                recipient=recipient,
                actor=actor,
                detail=detail,
                meta=meta or {},
                created_at=now,
                updated_at=now,
            )
        )
        await session.commit()
        return
    row.event = event
    row.title = title or row.title
    row.recipient = recipient
    row.actor = actor
    row.detail = detail
    row.meta = meta or {}
    row.created_at = now
    row.updated_at = now
    await session.execute(
        sa_delete(NotificationRead).where(NotificationRead.notification_id == row.id)
    )
    await session.commit()


def _upsert_event_sync(
    *,
    section: str,
    family: str,
    event: str,
    entity_id: str,
    title: str,
    recipient: str | None = None,
    detail: str | None = None,
    meta: dict | None = None,
) -> None:
    """Синхронный upsert для аварийных путей без живого event loop
    (SoftTimeLimitExceeded, poison-cap). Тот же контракт, что upsert_event."""
    from sqlalchemy import create_engine

    engine = create_engine(settings.database_url)
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO notifications"
                    " (recipient, actor, section, family, event, entity_id,"
                    "  title, detail, meta, created_at, updated_at)"
                    " VALUES (:recipient, NULL, :section, :family, :event,"
                    "  :entity_id, :title, :detail, CAST(:meta AS json), now(), now())"
                    " ON CONFLICT (section, entity_id, family) DO UPDATE SET"
                    "  event = excluded.event, title = excluded.title,"
                    "  recipient = excluded.recipient, detail = excluded.detail,"
                    "  meta = excluded.meta, created_at = now(), updated_at = now()"
                ),
                {
                    "recipient": recipient,
                    "section": section,
                    "family": family,
                    "event": event,
                    "entity_id": entity_id,
                    "title": title,
                    "detail": detail,
                    "meta": json.dumps(meta or {}, ensure_ascii=False),
                },
            )
            conn.execute(
                text(
                    "DELETE FROM notification_reads WHERE notification_id ="
                    " (SELECT id FROM notifications WHERE section = :section"
                    "  AND entity_id = :entity_id AND family = :family)"
                ),
                {"section": section, "entity_id": entity_id, "family": family},
            )
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# Эмиттеры (best-effort — никогда не роняют вызывающего)
# ---------------------------------------------------------------------------


def _file_title(fr: FileRecord) -> str:
    return fr.display_name or fr.original_name


async def notify_file_done(session: AsyncSession, file_id: str) -> None:
    """Материал обработан → событие в «Материалы»; если это был последний
    обрабатываемый материал кейса — ещё и «Кейс обработан» в «Кейсы»."""
    try:
        fr = await session.get(FileRecord, file_id)
        if fr is None:
            return
        case = await case_for_file(session, file_id)
        # recipient: загрузивший (owner). У опубликованных файлов owner=NULL —
        # broadcast; честная адресация появится с Keycloak (Б2).
        await upsert_event(
            session,
            section="materials",
            family="generation",
            event="done",
            entity_id=file_id,
            title=_file_title(fr),
            recipient=fr.owner,
            meta={"case_id": case[0]} if case else {},
        )
        if case:
            await _maybe_notify_case_done(session, case_id=case[0])
    except Exception as exc:  # noqa: BLE001 — уведомление не роняет пайплайн
        logger.warning("notify_file_done_failed", file_id=file_id, error=str(exc))
        await _safe_rollback(session)


async def notify_file_failed(session: AsyncSession, file_id: str, error: str) -> None:
    """Материал упал → событие «Ошибка» с причиной (никогда не молча)."""
    try:
        fr = await session.get(FileRecord, file_id)
        if fr is None:
            return
        case = await case_for_file(session, file_id)
        await upsert_event(
            session,
            section="materials",
            family="generation",
            event="failed",
            entity_id=file_id,
            title=_file_title(fr),
            recipient=fr.owner,
            detail=(error or "")[:500],
            meta={"case_id": case[0]} if case else {},
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("notify_file_failed_failed", file_id=file_id, error=str(exc))
        await _safe_rollback(session)


async def _maybe_notify_case_done(session: AsyncSession, case_id: str) -> None:
    """«Кейс обработан» — только когда ВСЕ его материалы дошли до терминального
    статуса и хотя бы один успешен. Догрузка материалов в кейс позже честно
    переэмитит то же событие (in-place, без наслоения)."""
    case = await session.get(CaseRecord, case_id)
    if case is None or not case.doc_ids:
        return
    statuses = (
        await session.execute(
            select(FileRecord.status).where(FileRecord.file_id.in_(case.doc_ids))
        )
    ).scalars().all()
    if not statuses or any(s.upper() not in _FILE_TERMINAL for s in statuses):
        return
    n_done = sum(1 for s in statuses if s.upper() == "DONE")
    if not n_done:
        return
    await upsert_event(
        session,
        section="cases",
        family="generation",
        event="done",
        entity_id=case_id,
        title=case.title,
        recipient=case.owner if case.sensitive else None,
        meta={"materials": n_done},
    )


async def notify_case_ready_if_done(session: AsyncSession, case_id: str) -> None:
    """«Кейс готов», если все его материалы уже терминальны (пункт 2).

    Best-effort обёртка над _maybe_notify_case_done для вызова из API-путей
    (создание кейса / прикрепление материалов). Нужна потому, что кейс из уже
    существующих (дедуп) материалов пайплайн не запускает — раньше по такому
    кейсу событие «готов» не приходило вовсе, и юзер его не видел в ленте.
    Если часть материалов ещё обрабатывается — не эмитит ничего (пайплайн
    добьёт «готов», когда закончит последний). По самим существующим
    материалам уведомления не создаются (дубликат не перезапускает обработку).
    """
    try:
        await _maybe_notify_case_done(session, case_id=case_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("notify_case_ready_failed", case_id=case_id, error=str(exc))
        await _safe_rollback(session)


async def notify_artifact_event(
    session: AsyncSession,
    *,
    artifact_id: str,
    document_id: str,
    kind: str,
    event: str,
    requested_by: str | None = None,
    detail: str | None = None,
) -> None:
    """Артефакт готов/упал → событие в «Артефакты». title — имя материала/кейса;
    вид артефакта фронт берёт из meta.kind (локализация на клиенте)."""
    try:
        title = await document_title(session, document_id)
        await upsert_event(
            session,
            section="artifacts",
            family="generation",
            event=event,
            entity_id=artifact_id,
            title=title,
            recipient=requested_by,
            detail=(detail or None) and detail[:500],
            meta={"document_id": document_id, "kind": kind},
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "notify_artifact_failed", artifact_id=artifact_id, error=str(exc)
        )
        await _safe_rollback(session)


def notify_artifact_failed_sync(artifact_id: str, error: str) -> None:
    """Аварийный синхронный вариант (loop уже мёртв): контекст артефакта
    дочитывается из БД тем же sync-подключением."""
    try:
        from sqlalchemy import create_engine

        engine = create_engine(settings.database_url)
        try:
            with engine.connect() as conn:
                row = conn.execute(
                    text(
                        "SELECT a.document_id, a.kind, a.requested_by,"
                        "  COALESCE(f.display_name, f.original_name, c.title,"
                        "           a.document_id) AS title"
                        " FROM artifacts a"
                        " LEFT JOIN files f ON f.file_id = a.document_id"
                        " LEFT JOIN cases c ON c.id = a.document_id"
                        " WHERE a.artifact_id = :aid"
                    ),
                    {"aid": artifact_id},
                ).first()
        finally:
            engine.dispose()
        if row is None:
            return
        _upsert_event_sync(
            section="artifacts",
            family="generation",
            event="failed",
            entity_id=artifact_id,
            title=row.title,
            recipient=row.requested_by,
            detail=error[:500],
            meta={"document_id": row.document_id, "kind": row.kind},
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "notify_artifact_failed_sync_failed", artifact_id=artifact_id, error=str(exc)
        )


def notify_file_failed_sync(file_id: str, error: str) -> None:
    """Аварийный синхронный вариант для файла (poison-cap путь)."""
    try:
        from sqlalchemy import create_engine

        engine = create_engine(settings.database_url)
        try:
            with engine.connect() as conn:
                row = conn.execute(
                    text(
                        "SELECT COALESCE(display_name, original_name) AS title,"
                        " owner FROM files WHERE file_id = :fid"
                    ),
                    {"fid": file_id},
                ).first()
        finally:
            engine.dispose()
        if row is None:
            return
        _upsert_event_sync(
            section="materials",
            family="generation",
            event="failed",
            entity_id=file_id,
            title=row.title,
            recipient=row.owner,
            detail=error[:500],
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("notify_file_failed_sync_failed", file_id=file_id, error=str(exc))


async def notify_case_privacy(
    session: AsyncSession,
    *,
    case_id: str,
    title: str,
    published: bool,
    actor: str,
) -> None:
    """Кейс стал общим/приватным — социальное событие, видно всем (broadcast).
    In-place по family="privacy": туда-сюда-переключения не спамят ленту."""
    try:
        await upsert_event(
            session,
            section="cases",
            family="privacy",
            event="published" if published else "privated",
            entity_id=case_id,
            title=title,
            recipient=None,
            actor=actor,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("notify_case_privacy_failed", case_id=case_id, error=str(exc))
        await _safe_rollback(session)


async def document_title(session: AsyncSession, document_id: str) -> str:
    """Имя сущности артефакта: материал (display_name) или кейс (title)."""
    if document_id.startswith("case-"):
        case = await session.get(CaseRecord, document_id)
        if case is not None:
            return case.title
    fr = await session.get(FileRecord, document_id)
    if fr is not None:
        return _file_title(fr)
    return document_id


async def _safe_rollback(session: AsyncSession) -> None:
    """Откатить сессию после сбоя эмита, не породив вторичную ошибку."""
    try:
        await session.rollback()
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Выборка ленты
# ---------------------------------------------------------------------------


def _visible_to(caller: str):  # noqa: ANN202 — SQLAlchemy expression
    return or_(
        NotificationRecord.recipient.is_(None),
        NotificationRecord.recipient == caller,
    )


def _read_join(caller: str):  # noqa: ANN202
    return and_(
        NotificationRead.notification_id == NotificationRecord.id,
        NotificationRead.user_key == caller,
    )


async def list_events(
    session: AsyncSession,
    caller: str,
    *,
    tab: str | None = None,
    limit: int = 30,
) -> list[tuple[NotificationRecord, bool]]:
    """События, видимые пользователю, новые сверху: (запись, прочитано)."""
    stmt = (
        select(NotificationRecord, NotificationRead.notification_id)
        .outerjoin(NotificationRead, _read_join(caller))
        .where(_visible_to(caller))
    )
    if tab:
        stmt = stmt.where(NotificationRecord.section == tab)
    stmt = stmt.order_by(NotificationRecord.created_at.desc()).limit(limit)
    rows = (await session.execute(stmt)).all()
    return [(rec, read_id is not None) for rec, read_id in rows]


async def unread_counts(session: AsyncSession, caller: str) -> dict[str, int]:
    """Непрочитанные события по вкладкам — бейдж колокольчика."""
    stmt = (
        select(NotificationRecord.section, func.count())
        .outerjoin(NotificationRead, _read_join(caller))
        .where(_visible_to(caller), NotificationRead.notification_id.is_(None))
        .group_by(NotificationRecord.section)
    )
    rows = (await session.execute(stmt)).all()
    counts = {s: 0 for s in SECTIONS}
    counts.update({section: int(n) for section, n in rows})
    return counts


async def mark_read(
    session: AsyncSession,
    caller: str,
    *,
    ids: list[int] | None = None,
    mark_all: bool = False,
) -> int:
    """Пометить прочитанными: конкретные ids или всё видимое (mark_all)."""
    if not ids and not mark_all:
        return 0
    stmt = (
        select(NotificationRecord.id)
        .outerjoin(NotificationRead, _read_join(caller))
        .where(_visible_to(caller), NotificationRead.notification_id.is_(None))
    )
    if ids:
        stmt = stmt.where(NotificationRecord.id.in_(ids))
    target = (await session.execute(stmt)).scalars().all()
    now = datetime.now(timezone.utc)
    for nid in target:
        session.add(NotificationRead(notification_id=nid, user_key=caller, read_at=now))
    if target:
        await session.commit()
    return len(target)


async def sweep_old_events(session: AsyncSession, days: int = 30) -> int:
    """Ретенция (janitor): события старше *days* дней удаляются вместе с
    отметками чтения. Возвращает число удалённых событий."""
    old_ids = (
        await session.execute(
            select(NotificationRecord.id).where(
                NotificationRecord.created_at
                < text(f"now() - interval '{int(days)} days'")
            )
        )
    ).scalars().all()
    if not old_ids:
        return 0
    await session.execute(
        sa_delete(NotificationRead).where(NotificationRead.notification_id.in_(old_ids))
    )
    await session.execute(
        sa_delete(NotificationRecord).where(NotificationRecord.id.in_(old_ids))
    )
    await session.commit()
    return len(old_ids)
