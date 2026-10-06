"""Presentation revisions and authorized binary assets. No public static mount."""

# ruff: noqa: B008
# FastAPI resolves dependency declarations in function defaults.
from __future__ import annotations

from copy import deepcopy
from typing import Literal

from fastapi import Depends, HTTPException, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from llm_wiki.api.deps import get_db, get_user_key
from llm_wiki.api.v1 import router
from llm_wiki.config import settings
from llm_wiki.storage.metadata import ArtifactRecord, ArtifactRevision, CaseRecord
from llm_wiki.storage.object_store import get_object_store
from llm_wiki.storage.visual_presentations import check_sources, progress


@router.get("/studio/capabilities")
async def capabilities(caller: str = Depends(get_user_key)) -> dict:
    return {
        "visual_presentations": settings.visual_presentations_enabled
        and settings.llm_provider == "openai",
        "presentation_revisions": True,
        "languages": ["ru", "kk", "en"],
    }


async def find_revision(session, record, language, revision, caller) -> ArtifactRevision:
    if language not in {"ru", "kk", "en"}:
        raise HTTPException(422, detail={"reason": "invalid_language"})
    stmt = select(ArtifactRevision).where(
        ArtifactRevision.artifact_id == record.artifact_id, ArtifactRevision.language == language
    )
    if revision is not None:
        stmt = stmt.where(ArtifactRevision.revision == revision)
    row = await session.scalar(stmt.order_by(ArtifactRevision.revision.desc()).limit(1))
    if row is None and record.kind == "presentation" and revision in {None, 1}:
        version = next((v for v in record.versions or [] if v.get("language") == language), None)
        if version and version.get("revision") is None:
            case = await session.get(CaseRecord, record.document_id)
            ids = version.get("source_doc_ids")
            if ids is None:
                ids = list(case.doc_ids or []) if case else [record.document_id]
            row = ArtifactRevision(
                artifact_id=record.artifact_id,
                language=language,
                revision=1,
                content=version["content"],
                sources=[{"id": id_} for id_ in ids],
                requested_by=record.requested_by,
                created_at=record.updated_at or record.created_at,
            )
    if row is None:
        raise HTTPException(
            404,
            detail={
                "reason": "revision_not_found" if revision is not None else "version_not_found"
            },
        )
    await check_sources(session, record, row.sources, caller, verify_content=False)
    return row


def public_content(row: ArtifactRevision) -> dict:
    content = deepcopy(row.content)
    content.pop("exports", None)
    for index, slide in enumerate(content.get("slides", []), 1):
        slide.pop("asset_key", None)
        slide.pop("thumbnail_key", None)
        if content.get("mode") == "visual":
            slide["image"] = (
                f"/api/v1/artifacts/{row.artifact_id}/assets/{row.language}/{row.revision}/{index}"
            )
            slide["thumbnail"] = slide["image"] + "?variant=thumb"
    return content


async def revision_detail(session, record, language, revision, caller) -> dict:
    generation = await progress(session, record)
    if generation is None and record.kind == "presentation" and record.generation_context:
        generation = {
            "language": record.generation_context.get("language"),
            "status": record.status,
            "stage": "planning" if record.started_at else "queued",
            "slides_done": 0,
            "slides_total": 0,
        }
    versions, stale = [], False
    try:
        row = await find_revision(session, record, language, revision, caller)
    except HTTPException as exc:
        if (
            exc.status_code != 404
            or revision is not None
            or not isinstance(exc.detail, dict)
            or exc.detail.get("reason") != "version_not_found"
            or not generation
            or generation["language"] != language
        ):
            raise
    else:
        stale = await check_sources(session, record, row.sources, caller)
        content = public_content(row)
        if record.kind == "presentation":
            from llm_wiki.agents.presentation_layout import presentation_layout

            await session.commit()
            content["layout"] = await run_in_threadpool(presentation_layout, content)
        versions = [
            {
                "language": language,
                "revision": row.revision,
                "content": content,
                "source_doc_ids": [s["id"] for s in row.sources],
            }
        ]
    return {
        "artifact_id": record.artifact_id,
        "kind": record.kind,
        "status": record.status,
        "error": record.error,
        "versions": versions,
        "generation": generation,
        "stale": stale,
    }


@router.get("/artifacts/{artifact_id}/revisions")
async def revisions(
    artifact_id: str,
    language: str = "ru",
    session: AsyncSession = Depends(get_db),
    caller: str = Depends(get_user_key),
) -> list[dict]:
    from llm_wiki.api.v1.artifacts import _check_document_access

    record = await session.get(ArtifactRecord, artifact_id)
    if record is None:
        raise HTTPException(404)
    await _check_document_access(session, record.document_id, caller)
    if language not in {"ru", "kk", "en"}:
        raise HTTPException(422, detail={"reason": "invalid_language"})
    result = []
    for row in await session.scalars(
        select(ArtifactRevision)
        .where(ArtifactRevision.artifact_id == artifact_id, ArtifactRevision.language == language)
        .order_by(ArtifactRevision.revision.desc())
        .limit(10)
    ):
        try:
            await check_sources(session, record, row.sources, caller, verify_content=False)
        except HTTPException:
            continue  # Do not reveal now-inaccessible historical sources.
        result.append(
            {
                "revision": row.revision,
                "created_at": row.created_at.isoformat(),
                "requested_by": row.requested_by,
                "slide_count": len(row.content.get("slides", [])),
            }
        )
    if not result and record.kind == "presentation":
        try:
            row = await find_revision(session, record, language, None, caller)
        except HTTPException:
            pass
        else:
            result.append(
                {
                    "revision": row.revision,
                    "created_at": row.created_at.isoformat(),
                    "requested_by": row.requested_by,
                    "slide_count": len(row.content.get("slides", [])),
                }
            )
    return result


async def _bytes(session, key: str) -> bytes:
    await session.commit()  # Release the DB connection before object-store I/O.
    try:
        return await run_in_threadpool(lambda: get_object_store().get_bytes(key))
    except Exception as exc:
        raise HTTPException(503, detail={"reason": "asset_unavailable"}) from exc


@router.get("/artifacts/{artifact_id}/assets/{language}/{revision}/{index}")
async def asset(
    artifact_id: str,
    language: str,
    revision: int,
    index: int,
    session: AsyncSession = Depends(get_db),
    caller: str = Depends(get_user_key),
    variant: Literal["full", "thumb"] = "full",
) -> Response:
    record = await session.get(ArtifactRecord, artifact_id)
    if record is None or record.kind != "presentation_visual":
        raise HTTPException(404)
    row = await find_revision(session, record, language, revision, caller)
    slides = row.content.get("slides", [])
    if not 1 <= index <= len(slides):
        raise HTTPException(404)
    slide = slides[index - 1]
    data = await _bytes(
        session, (slide.get("thumbnail_key") if variant == "thumb" else None) or slide["asset_key"]
    )
    return Response(data, media_type="image/png", headers={"Cache-Control": "private, no-store"})


async def export_revision(session, record, language, revision, fmt, caller) -> Response:
    if fmt not in {"pdf", "pptx"}:
        raise HTTPException(422, detail={"reason": "export_format_invalid"})
    row = await find_revision(session, record, language, revision, caller)
    key = (row.content.get("exports") or {}).get(fmt)
    if not key:
        raise HTTPException(409, detail={"reason": "export_not_ready"})
    name = f"presentation-visual-{record.artifact_id[:8]}-{language}-r{row.revision}.{fmt}"
    data = await _bytes(session, key)
    from llm_wiki.agents.artifacts_export import MEDIA_TYPES

    return Response(
        data,
        media_type=MEDIA_TYPES[fmt],
        headers={
            "Cache-Control": "private, no-store",
            "Content-Disposition": f'attachment; filename="{name}"',
        },
    )
