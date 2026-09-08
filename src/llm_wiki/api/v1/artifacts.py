"""Studio artifact endpoints — real generation + storage (replaces the mocks).

Contract expected by the frontend ``StudioColumn``:
- ``GET  /artifacts?document_id=`` → [{artifact_id, kind, status, created_at}]
- ``GET  /artifacts/{id}?language=`` → {artifact_id, kind, versions:[{language, content}]}
- ``POST /studio/generate`` {kind, document_id, language, source_doc_ids?} → {artifact_id, kind, status}
- ``POST /cards/generate``  {document_id, languages:[..], source_doc_ids?} → {artifact_id, kind:"card", status}
- ``POST /images/generate`` {document_id, language, source_doc_ids?} → {artifact_id, kind:"infographic", status}
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from fastapi import Depends, HTTPException, Response
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.agents.artifacts import ArtifactError, _title_and_slugs
from llm_wiki.agents.artifacts_export import ExportError, export_artifact, supported_formats
from llm_wiki.api.deps import get_db, get_user_key
from llm_wiki.api.v1 import router
from llm_wiki.storage import artifacts_store
from llm_wiki.storage.metadata import CaseRecord, FileRecord


async def _check_document_access(session: AsyncSession, document_id: str, caller: str) -> None:
    record = await session.get(CaseRecord, document_id)
    if record is None:
        record = await session.get(FileRecord, document_id)
    if record is None or (record.sensitive and record.owner != caller):
        raise HTTPException(status_code=404, detail="Источник не найден")


def _body_sources(body: dict[str, Any]) -> list[str] | None:
    ids = body.get("source_doc_ids")
    if ids is None:
        return None
    if not isinstance(ids, list) or any(not isinstance(i, str) or not i.strip() for i in ids):
        raise HTTPException(status_code=422, detail="Некорректный список источников")
    if not ids:
        raise HTTPException(status_code=422, detail="Выберите хотя бы один источник для генерации")
    return list(dict.fromkeys(ids))


async def _resolve_sources(
    session: AsyncSession, document_id: str, selected: list[str] | None, caller: str | None,
) -> list[str]:
    case = await session.get(CaseRecord, document_id)
    file = None if case is not None else await session.get(FileRecord, document_id)
    parent = case if case is not None else file
    if parent is not None and caller is not None and parent.sensitive and parent.owner != caller:
        raise HTTPException(status_code=404, detail="Источник не найден")
    members = list(case.doc_ids or []) if case is not None else [document_id]
    ids = members if selected is None else selected
    if selected is not None and (not ids or not set(ids).issubset(members)):
        raise HTTPException(status_code=422, detail="Выбранные источники отсутствуют в кейсе. Обновите список материалов.")
    if len(ids) > 200:
        raise HTTPException(status_code=422, detail="Выберите не более 200 источников для одной генерации.")
    for file_id in ids:
        source = await session.get(FileRecord, file_id)
        if source is None or source.status != "DONE":
            raise HTTPException(status_code=422, detail="Источник пока без содержимого — материалы ещё обрабатываются или были удалены. Дождитесь обработки и попробуйте снова.")
        if caller is not None and source.sensitive and source.owner != caller:
            raise HTTPException(status_code=404, detail="Источник не найден")
    return list(dict.fromkeys(ids))


@router.get("/artifacts")
async def list_artifacts(
    document_id: str | None = None, session: AsyncSession = Depends(get_db),
    language: str = "ru", caller: str = Depends(get_user_key),
) -> list[dict[str, Any]]:
    if not document_id:
        return []
    await _check_document_access(session, document_id, caller)
    rows = await artifacts_store.list_artifacts(session, document_id)
    return [artifacts_store.serialize_summary(r, language) for r in rows]


@router.get("/artifacts/{artifact_id}")
async def get_artifact(
    artifact_id: str,
    language: str = "ru",  # noqa: ARG001 — frontend passes it; we return all versions
    session: AsyncSession = Depends(get_db),
    caller: str = Depends(get_user_key),
) -> dict[str, Any]:
    record = await artifacts_store.get_artifact(session, artifact_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    await _check_document_access(session, record.document_id, caller)
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
    caller: str = Depends(get_user_key),
) -> Response:
    """Stream the rendered artifact as a downloadable file.

    Reached by ``window.open`` (browser navigation), so it carries no auth
    header — fine while AUTH runs in demo mode. The POST sibling below validates
    the request under auth; this route only serves the bytes.
    """
    record = await artifacts_store.get_artifact(session, artifact_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    await _check_document_access(session, record.document_id, caller)
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
    await _check_document_access(session, record.document_id, _caller)
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
    generation_id: str,
) -> None:
    """Report an enqueue failure only while this attempt still owns the row."""
    if not await artifacts_store.mark_failed(session, artifact_id, error, generation_id=generation_id):
        return
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


async def _reject_empty_source(session: AsyncSession, document_id: str, source_doc_ids: list[str] | None = None) -> None:
    """Fail-fast (QA): (пере)генерация по заведомо пустому источнику — 422
    СРАЗУ, до создания pending-строки и постановки в очередь.

    Юзер-сценарий: удалил все материалы кейса → жмёт «Пересоздать» → раньше
    генерация честно стартовала, минуту спустя падала фоном, а прошлый готовый
    артефакт при этом перезатирался статусом failed. Теперь ошибка приходит в
    момент клика, а раз генерация даже не начинается — старый артефакт
    остаётся «Готово» и ничего не теряется.

    Проверка та же, какой видит источники сама генерация (_title_and_slugs):
    кейс без материалов / материалы без вики-страниц (ещё обрабатываются или
    упали) / несуществующий документ.
    """
    from llm_wiki.storage.metadata import CaseRecord

    case = await session.get(CaseRecord, document_id)
    if case is not None and not (case.doc_ids or []):
        raise HTTPException(
            status_code=422,
            detail=(
                "В кейсе нет материалов — добавьте хотя бы один материал, "
                "чтобы создать артефакт."
            ),
        )
    try:
        _title, slugs = await _title_and_slugs(session, document_id, source_doc_ids)
    except ArtifactError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not slugs:
        raise HTTPException(
            status_code=422,
            detail=(
                "Источник пока без содержимого — материалы ещё обрабатываются "
                "или были удалены. Дождитесь обработки и попробуйте снова."
            ),
        )


# A pending artifact younger than this is considered a LIVE generation —
# repeat clicks return the same id instead of enqueuing another heavy task.
# Above the worker deadline (480 s) so a running generation is never doubled;
# expired/queued-and-lost rows age past it and become regenerable (the janitor
# usually fails them sooner).
_PENDING_DEDUP_WINDOW_S = 600


async def _start_generation(
    session: AsyncSession, kind: str, document_id: str, language: str,
    requested_by: str | None = None,
    source_doc_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Create a pending artifact and enqueue background generation.

    Heavy artifacts (LLM reports, gpt-image-1 infographics) can run for minutes,
    so we move them off the request path: the client gets an id immediately and
    polls ``GET /artifacts/{id}`` until ``status`` is ``ready``/``failed``. If the
    broker is unreachable, fail promptly and retain the last successful version.

    Dedup: during the 2026-08-20 incident stuck «pending» rows made users spam
    the generate button, each click enqueuing another multi-minute LLM task.
    A fresh pending row now short-circuits to the same artifact id.
    """
    from datetime import datetime, timedelta, timezone

    # Fail-fast: пустой источник отклоняется ДО каких-либо записей — старый
    # готовый артефакт не переводится в pending/failed и не теряется.
    source_doc_ids = await _resolve_sources(session, document_id, source_doc_ids, requested_by)
    await _reject_empty_source(session, document_id, source_doc_ids)
    # Serialize concurrent starts, including the first insert for this kind.
    await session.execute(text("SELECT pg_advisory_xact_lock(hashtext(:key))"), {"key": f"artifact:{document_id}:{kind}"})

    existing = await artifacts_store.find_by_kind(session, document_id, kind)
    if existing is not None and existing.status == "pending":
        ts = existing.updated_at or existing.created_at
        if ts is not None and ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts is not None and datetime.now(timezone.utc) - ts < timedelta(
            seconds=_PENDING_DEDUP_WINDOW_S
        ):
            context = existing.generation_context or {}
            if context and (set(context.get("source_doc_ids", [])) != set(source_doc_ids) or context.get("language") != language):
                raise HTTPException(status_code=409, detail="Артефакт уже создаётся по другому набору источников или языку. Дождитесь завершения.")
            return {
                "artifact_id": existing.artifact_id,
                "kind": kind,
                "status": "pending",
            }

    context = {"id": uuid4().hex, "source_doc_ids": source_doc_ids, "language": language}
    record = await artifacts_store.create_pending_artifact(
        session, document_id=document_id, kind=kind, requested_by=requested_by, generation_context=context
    )
    try:
        from llm_wiki.orchestrator.tasks import generate_artifact

        # expires: a message not picked up within 30 min is dropped by the
        # worker — after an outage the queue drains instantly instead of
        # burning LLM budget on generations nobody is waiting for anymore
        # (the janitor has already failed their rows).
        generate_artifact.apply_async(
            args=(record.artifact_id, document_id, kind, language),
            kwargs={"generation_id": context["id"]},
            expires=1800,
        )
    except Exception as exc:  # noqa: BLE001 — do not run paid work in a failed HTTP enqueue
        error = "Очередь генерации временно недоступна. Попробуйте ещё раз позже."
        await _notify_artifact_failed(
            session, artifact_id=record.artifact_id, document_id=document_id,
            kind=kind, requested_by=requested_by, error=error, generation_id=context["id"],
        )
        raise HTTPException(status_code=503, detail=error) from exc
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
    return await _start_generation(session, kind, document_id, language, requested_by=caller, source_doc_ids=_body_sources(body))


@router.post("/cards/generate", status_code=202)
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
    # Cards use the same durable queue and deduplication as other artifact kinds.
    return await _start_generation(
        session, "card", document_id, language, requested_by=caller, source_doc_ids=_body_sources(body)
    )


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
    return await _start_generation(session, "infographic", document_id, language, requested_by=caller, source_doc_ids=_body_sources(body))
