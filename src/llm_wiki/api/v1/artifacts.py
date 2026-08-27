"""Studio artifact endpoints — real generation + storage (replaces the mocks).

Contract expected by the frontend ``StudioColumn``:
- ``GET  /artifacts?document_id=`` → [{artifact_id, kind, status, created_at}]
- ``GET  /artifacts/{id}?language=`` → {artifact_id, kind, versions:[{language, content}]}
- ``POST /studio/generate`` {kind, document_id, language} → {artifact_id, kind, content}  (report/test/presentation)
- ``POST /cards/generate``  {document_id, languages:[..]} → {artifact_id, kind:"card"}
- ``POST /images/generate`` {document_id, language, title} → {artifact_id, kind:"infographic", svg}
"""

from __future__ import annotations

from typing import Any

from fastapi import Depends, HTTPException, Response
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.agents.artifacts import ArtifactError, generate_content
from llm_wiki.agents.artifacts_export import ExportError, export_artifact, supported_formats
from llm_wiki.api.deps import get_db, get_user_key
from llm_wiki.api.v1 import router
from llm_wiki.storage import artifacts_store


@router.get("/artifacts")
async def list_artifacts(
    document_id: str | None = None, session: AsyncSession = Depends(get_db)
) -> list[dict[str, Any]]:
    if not document_id:
        return []
    rows = await artifacts_store.list_artifacts(session, document_id)
    return [artifacts_store.serialize_summary(r) for r in rows]


@router.get("/artifacts/{artifact_id}")
async def get_artifact(
    artifact_id: str,
    language: str = "ru",  # noqa: ARG001 — frontend passes it; we return all versions
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    record = await artifacts_store.get_artifact(session, artifact_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    return artifacts_store.serialize_detail(record)


def _version_content(record: Any, language: str) -> dict[str, Any]:
    """Pick the ``language`` version's content (fall back to the first stored)."""
    versions = record.versions or []
    for v in versions:
        if isinstance(v, dict) and v.get("language") == language:
            return v.get("content") or {}
    for v in versions:
        if isinstance(v, dict):
            return v.get("content") or {}
    return {}


@router.get("/artifacts/{artifact_id}/export")
async def export_artifact_file(
    artifact_id: str,
    format: str = "pdf",
    language: str = "ru",
    session: AsyncSession = Depends(get_db),
) -> Response:
    """Stream the rendered artifact as a downloadable file.

    Reached by ``window.open`` (browser navigation), so it carries no auth
    header — fine while AUTH runs in demo mode. The POST sibling below validates
    the request under auth; this route only serves the bytes.
    """
    record = await artifacts_store.get_artifact(session, artifact_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    try:
        data, media_type = export_artifact(
            record.kind, _version_content(record, language), format
        )
    except ExportError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    filename = f"{record.kind}-{artifact_id[:8]}.{format}"
    return Response(
        content=data,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/artifacts/{artifact_id}/export")
async def export_artifact_prepare(
    artifact_id: str,
    body: dict[str, Any],
    session: AsyncSession = Depends(get_db),
    _caller: str = Depends(get_user_key),
) -> dict[str, Any]:
    """Validate an export request (kind/format) so the UI can toast on error.

    Returns ``{}`` (no ``url``) so the frontend downloads via the GET route.
    """
    record = await artifacts_store.get_artifact(session, artifact_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    fmt = str(body.get("format") or "")
    if fmt not in supported_formats(record.kind):
        raise HTTPException(
            status_code=400, detail=f"Cannot export {record.kind!r} as {fmt!r}"
        )
    return {}


async def _notify_artifact_failed(
    session: AsyncSession,
    *,
    artifact_id: str,
    document_id: str,
    kind: str,
    requested_by: str | None,
    error: str,
) -> None:
    """Пометить артефакт упавшим И положить «ошибка» в ленту — синхронные пути
    (карточки, inline-фолбэк) сами не проходят через Celery-таск, где это
    делается. Так ошибка «нет материалов» видна и в тосте, и в колокольчике."""
    await artifacts_store.mark_failed(session, artifact_id, error)
    from llm_wiki.storage import notifications as notif

    await notif.notify_artifact_event(
        session,
        artifact_id=artifact_id,
        document_id=document_id,
        kind=kind,
        event="failed",
        requested_by=requested_by,
        detail=error,
    )


async def _generate_and_store(
    session: AsyncSession, kind: str, document_id: str, language: str,
    requested_by: str | None = None,
) -> tuple[str, dict[str, Any]]:
    from llm_wiki.llm.client import LLMClient

    llm = LLMClient()
    try:
        content = await generate_content(
            session, llm, kind=kind, document_id=document_id, language=language
        )
    finally:
        await llm.aclose()
    record = await artifacts_store.upsert_artifact(
        session, document_id=document_id, kind=kind, language=language, content=content
    )
    # Б1: синхронный путь (карточки; и inline-фолбэк при недоступном брокере)
    # НЕ проходит через Celery-таск, где эмитится «артефакт готов» — поэтому
    # событие в ленту нужно эмитить здесь, иначе по карточкам уведомления нет.
    from llm_wiki.storage import notifications as notif

    await notif.notify_artifact_event(
        session,
        artifact_id=record.artifact_id,
        document_id=document_id,
        kind=kind,
        event="done",
        requested_by=requested_by,
    )
    return record.artifact_id, content


# A pending artifact younger than this is considered a LIVE generation —
# repeat clicks return the same id instead of enqueuing another heavy task.
# Above the worker deadline (480 s) so a running generation is never doubled;
# expired/queued-and-lost rows age past it and become regenerable (the janitor
# usually fails them sooner).
_PENDING_DEDUP_WINDOW_S = 600


async def _start_generation(
    session: AsyncSession, kind: str, document_id: str, language: str,
    requested_by: str | None = None,
) -> dict[str, Any]:
    """Create a pending artifact and enqueue background generation.

    Heavy artifacts (LLM reports, gpt-image-1 infographics) can run for minutes,
    so we move them off the request path: the client gets an id immediately and
    polls ``GET /artifacts/{id}`` until ``status`` is ``ready``/``failed``. If the
    broker is unreachable (e.g. dev without a worker) we fall back to generating
    synchronously so the feature still works.

    Dedup: during the 2026-08-20 incident stuck «pending» rows made users spam
    the generate button, each click enqueuing another multi-minute LLM task.
    A fresh pending row now short-circuits to the same artifact id.
    """
    from datetime import datetime, timedelta, timezone

    existing = await artifacts_store.find_by_kind(session, document_id, kind)
    if existing is not None and existing.status == "pending":
        ts = existing.updated_at or existing.created_at
        if ts is not None and ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts is not None and datetime.now(timezone.utc) - ts < timedelta(
            seconds=_PENDING_DEDUP_WINDOW_S
        ):
            return {
                "artifact_id": existing.artifact_id,
                "kind": kind,
                "status": "pending",
            }

    record = await artifacts_store.create_pending_artifact(
        session, document_id=document_id, kind=kind, requested_by=requested_by
    )
    try:
        from llm_wiki.orchestrator.tasks import generate_artifact

        # expires: a message not picked up within 30 min is dropped by the
        # worker — after an outage the queue drains instantly instead of
        # burning LLM budget on generations nobody is waiting for anymore
        # (the janitor has already failed their rows).
        generate_artifact.apply_async(
            args=(record.artifact_id, document_id, kind, language),
            expires=1800,
        )
    except Exception:  # noqa: BLE001 — broker down: generate inline as a fallback
        try:
            await _generate_and_store(
                session, kind, document_id, language, requested_by=requested_by
            )
        except ArtifactError as exc:
            await _notify_artifact_failed(
                session, artifact_id=record.artifact_id, document_id=document_id,
                kind=kind, requested_by=requested_by, error=str(exc),
            )
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"artifact_id": record.artifact_id, "kind": kind, "status": "ready"}
    return {"artifact_id": record.artifact_id, "kind": kind, "status": "pending"}


@router.post("/studio/generate", status_code=202)
async def studio_generate(
    body: dict[str, Any],
    session: AsyncSession = Depends(get_db),
    caller: str = Depends(get_user_key),
) -> dict[str, Any]:
    kind = str(body.get("kind") or "")
    document_id = str(body.get("document_id") or "")
    language = str(body.get("language") or "ru")
    if kind not in ("report", "test", "presentation"):
        raise HTTPException(status_code=400, detail=f"Unsupported kind for /studio/generate: {kind!r}")
    if not document_id:
        raise HTTPException(status_code=400, detail="document_id is required")
    return await _start_generation(session, kind, document_id, language, requested_by=caller)


@router.post("/cards/generate")
async def cards_generate(
    body: dict[str, Any],
    session: AsyncSession = Depends(get_db),
    caller: str = Depends(get_user_key),
) -> dict[str, Any]:
    document_id = str(body.get("document_id") or "")
    langs = body.get("languages")
    language = str(langs[0]) if isinstance(langs, list) and langs else "ru"
    if not document_id:
        raise HTTPException(status_code=400, detail="document_id is required")
    # Заводим pending-строку заранее — на провале есть artifact_id для «ошибки»
    # в ленте (и клик из уведомления ведёт на упавший артефакт).
    record = await artifacts_store.create_pending_artifact(
        session, document_id=document_id, kind="card", requested_by=caller
    )
    try:
        artifact_id, _ = await _generate_and_store(
            session, "card", document_id, language, requested_by=caller
        )
    except ArtifactError as exc:
        await _notify_artifact_failed(
            session, artifact_id=record.artifact_id, document_id=document_id,
            kind="card", requested_by=caller, error=str(exc),
        )
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"artifact_id": artifact_id, "kind": "card"}


@router.post("/images/generate", status_code=202)
async def images_generate(
    body: dict[str, Any],
    session: AsyncSession = Depends(get_db),
    caller: str = Depends(get_user_key),
) -> dict[str, Any]:
    document_id = str(body.get("document_id") or "")
    language = str(body.get("language") or "ru")
    if not document_id:
        raise HTTPException(status_code=400, detail="document_id is required")
    # gpt-image-1 generation is the slowest artifact — always run it async and
    # let the client poll GET /artifacts/{id} for the content.
    return await _start_generation(session, "infographic", document_id, language, requested_by=caller)
