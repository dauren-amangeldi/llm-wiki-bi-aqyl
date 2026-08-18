"""Ops monitoring — что сейчас на генерации, кто отправил, статусы, очереди.

Один JSON-эндпоинт, на который наводится корпоративная Grafana (JSON/Infinity
datasource, заголовок ``X-Ops-Token``) — своей ops-страницы во фронте нет и не
планируется, контейнеры (Prometheus/Flower) в проект не закладываем.

Отдаёт два вида работ:
- **артефакты** (Celery ``generate_artifact``): pending → running → ready/failed,
  с ``requested_by``/``started_at``/``finished_at``;
- **файлы** (ингест ``process_file``): активные + недавние, владельца и статус
  ведёт ``FileRecord``.

Плюс глубина очередей брокера (Redis LLEN по ``ingest``/``artifacts``/``light``)
— видно, сколько работ ждёт слота, ещё до того как воркер их взял.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Literal

import structlog
from fastapi import Depends, Header, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.api.deps import get_db
from llm_wiki.api.v1 import router
from llm_wiki.config import settings
from llm_wiki.storage.metadata import ArtifactRecord, FileRecord

logger = structlog.get_logger(__name__)

# Статусы файлов, которые считаются «ещё в работе» (см. CASE_PROCESSING_STATUSES
# на фронте + пайплайн state_history).
_FILE_ACTIVE = {"received", "stored", "searched", "written", "linted", "logged", "processing", "pending"}

_QUEUES = ("ingest", "artifacts", "light")


def _require_ops_access(x_ops_token: str | None = Header(default=None)) -> None:
    """Grafana шлёт X-Ops-Token; без настроенного токена эндпоинт открыт только
    в demo-режиме (auth off) — на проде без токена всегда 403."""
    if settings.ops_token:
        if x_ops_token != settings.ops_token:
            raise HTTPException(status_code=403, detail="Invalid ops token")
        return
    if settings.auth_enabled:
        raise HTTPException(status_code=403, detail="OPS_TOKEN is not configured")


def _queue_depths() -> dict[str, int | None]:
    """LLEN по очередям Celery в Redis; None — брокер недоступен."""
    try:
        import redis

        r = redis.Redis.from_url(settings.redis_url, socket_timeout=2)
        return {q: int(r.llen(q)) for q in _QUEUES}
    except Exception as exc:  # noqa: BLE001 — мониторинг не должен падать целиком
        logger.warning("ops_queue_depth_failed", error=str(exc))
        return {q: None for q in _QUEUES}


def _duration_s(start: datetime | None, end: datetime | None) -> float | None:
    if start is None:
        return None
    return round(((end or datetime.now(timezone.utc)) - start).total_seconds(), 1)


@router.get("/ops/generations")
async def list_generations(
    limit: int = Query(50, ge=1, le=200, description="Сколько последних работ вернуть"),
    session: AsyncSession = Depends(get_db),
    _access: None = Depends(_require_ops_access),
) -> dict[str, Any]:
    """Активные и недавние генерации + глубина очередей.

    ``items`` отсортированы: сначала активные (pending/обработка), затем
    завершённые по убыванию времени. Статусы: ``queued`` (ждёт воркера),
    ``running``, ``ready``, ``failed``.
    """
    artifacts = (
        await session.scalars(
            select(ArtifactRecord).order_by(ArtifactRecord.updated_at.desc()).limit(limit)
        )
    ).all()
    files = (
        await session.scalars(
            select(FileRecord).order_by(FileRecord.updated_at.desc()).limit(limit)
        )
    ).all()

    items: list[dict[str, Any]] = []
    for a in artifacts:
        status = (
            "running" if a.status == "pending" and a.started_at is not None
            else "queued" if a.status == "pending"
            else a.status  # ready | failed
        )
        items.append(
            {
                "type": "artifact",
                "id": a.artifact_id,
                "what": a.kind,
                "document_id": a.document_id,
                "requested_by": a.requested_by,
                "status": status,
                "error": a.error,
                "created_at": a.created_at.isoformat() if a.created_at else None,
                "started_at": a.started_at.isoformat() if a.started_at else None,
                "finished_at": a.finished_at.isoformat() if a.finished_at else None,
                "duration_s": _duration_s(a.started_at, a.finished_at),
            }
        )
    for f in files:
        low = (f.status or "").lower()
        status = (
            "running" if low in _FILE_ACTIVE
            else "failed" if low == "failed"
            else "ready"
        )
        items.append(
            {
                "type": "file",
                "id": f.file_id,
                "what": f.original_name,
                "document_id": f.file_id,
                "requested_by": f.owner,
                "status": status,
                "error": f.error,
                "created_at": f.created_at.isoformat() if f.created_at else None,
                "started_at": f.created_at.isoformat() if f.created_at else None,
                "finished_at": (
                    f.updated_at.isoformat() if f.updated_at and status in ("ready", "failed") else None
                ),
                "duration_s": _duration_s(
                    f.created_at, f.updated_at if status in ("ready", "failed") else None
                ),
            }
        )

    active_rank = {"running": 0, "queued": 1, "failed": 2, "ready": 3}
    items.sort(
        key=lambda it: (
            active_rank.get(str(it["status"]), 4),
            -(datetime.fromisoformat(it["created_at"]).timestamp() if it["created_at"] else 0),
        )
    )

    return {
        "queues": _queue_depths(),
        "items": items[:limit],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ── Реальное состояние Celery + аварийная чистка очередей ────────────────────


def _celery_snapshot() -> dict[str, Any]:
    """Живой опрос воркеров (не логи и не БД): что реально выполняется (active),
    что взято с брокера, но ждёт слота (reserved), и что отложено (scheduled).
    Пустой dict воркеров = ни один воркер не ответил за таймаут."""
    from llm_wiki.orchestrator.tasks import celery_app

    insp = celery_app.control.inspect(timeout=3.0)

    def _short(tasks: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        out = []
        for t in tasks or []:
            out.append(
                {
                    "id": t.get("id"),
                    "task": (t.get("name") or "").rsplit(".", 1)[-1],
                    "args": str(t.get("args") or "")[:120],
                    "queue": ((t.get("delivery_info") or {}).get("routing_key")),
                    "started_at": t.get("time_start"),
                }
            )
        return out

    active = insp.active() or {}
    reserved = insp.reserved() or {}
    scheduled = insp.scheduled() or {}
    workers = sorted({*active, *reserved, *scheduled})
    return {
        "workers": {
            w: {
                "active": _short(active.get(w)),
                "reserved": _short(reserved.get(w)),
                "scheduled_count": len(scheduled.get(w) or []),
            }
            for w in workers
        },
    }


@router.get("/ops/celery")
async def celery_state(
    _access: None = Depends(_require_ops_access),
) -> dict[str, Any]:
    """Что происходит ВНУТРИ Celery прямо сейчас (по каждому воркеру).

    Дополняет /ops/generations: там очередь (Redis LLEN) и результат (БД), здесь
    — задачи, уже взятые воркерами. Вместе — полная картина конвейера.
    """
    snapshot = await asyncio.to_thread(_celery_snapshot)
    snapshot["queues"] = _queue_depths()
    snapshot["generated_at"] = datetime.now(timezone.utc).isoformat()
    return snapshot


class PurgeRequest(BaseModel):
    """Что чистим. Пустой queues = все четыре."""

    queues: list[Literal["ingest", "artifacts", "light", "celery"]] | None = None
    # Ещё и снять задачи, которые УЖЕ выполняются (revoke terminate) — по
    # умолчанию нет: обычно достаточно вычистить хвост очереди.
    terminate: bool = False


@router.post("/ops/queues/purge")
async def purge_queues(
    body: PurgeRequest,
    session: AsyncSession = Depends(get_db),
    _access: None = Depends(_require_ops_access),
) -> dict[str, Any]:
    """Аварийная чистка застрявших генераций (Celery иногда «залипает»).

    1. Удаляет сообщения из выбранных очередей брокера (queue_purge).
    2. При ``terminate`` — снимает и уже выполняющиеся задачи (revoke).
    3. Помечает зависшие pending-артефакты и «вечно обрабатывающиеся» файлы
       как failed с понятной причиной — карточки в UI перестают крутиться,
       пользователь может запустить генерацию заново.
    """
    from sqlalchemy import update as sa_update

    from llm_wiki.orchestrator.tasks import celery_app
    from llm_wiki.storage.metadata import ArtifactRecord, FileRecord

    queues = body.queues or ["ingest", "artifacts", "light", "celery"]

    def _purge() -> dict[str, int]:
        purged: dict[str, int] = {q: 0 for q in queues}
        try:
            with celery_app.connection_for_write() as conn:
                for q in queues:
                    try:
                        purged[q] = int(conn.default_channel.queue_purge(q) or 0)
                    except Exception:  # noqa: BLE001 — очереди могло не быть
                        purged[q] = 0
        except Exception as exc:  # noqa: BLE001 — брокер лёг: чистить нечего,
            # но пометка зависших записей в БД ниже всё равно отработает.
            logger.warning("ops_purge_broker_unreachable", error=str(exc))
        return purged

    purged = await asyncio.to_thread(_purge)

    revoked: list[str] = []
    if body.terminate:
        def _revoke() -> list[str]:
            insp = celery_app.control.inspect(timeout=3.0)
            ids = [
                t.get("id")
                for tasks in (insp.active() or {}).values()
                for t in tasks or []
                if t.get("id")
            ]
            for tid in ids:
                celery_app.control.revoke(tid, terminate=True)
            return ids

        revoked = await asyncio.to_thread(_revoke)

    note = "Отменено оператором (очистка очередей)"
    art = await session.execute(
        sa_update(ArtifactRecord)
        .where(ArtifactRecord.status == "pending")
        .values(status="failed", error=note, finished_at=datetime.now(timezone.utc))
    )
    files = await session.execute(
        sa_update(FileRecord)
        .where(func.lower(FileRecord.status).in_(list(_FILE_ACTIVE)))
        .values(status="FAILED", error=note)
    )
    await session.commit()

    logger.warning(
        "ops_queues_purged",
        queues=queues,
        purged=purged,
        revoked=len(revoked),
        artifacts_failed=art.rowcount,
        files_failed=files.rowcount,
    )
    return {
        "purged": purged,
        "revoked": revoked,
        "artifacts_marked_failed": art.rowcount,
        "files_marked_failed": files.rowcount,
    }
