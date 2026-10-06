"""Visual attempt admission, revision snapshots and live progress."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from fastapi import HTTPException
from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import undefer

from llm_wiki.agents.visual_presentation import LANGUAGES, PROMPT_VERSION, digest
from llm_wiki.config import settings
from llm_wiki.storage.metadata import (
    ArtifactRecord,
    ArtifactRevision,
    CaseRecord,
    FileRecord,
    VisualJob,
    VisualUnit,
)


def now() -> datetime:
    return datetime.now(UTC)


def source_hash(row: FileRecord) -> str:
    return digest({"id": row.file_id, "text": row.extracted_text, "raw_key": row.raw_key})


async def snapshot(session: AsyncSession, document_id: str, ids: list[str]) -> dict:
    case = await session.get(CaseRecord, document_id)
    length = await session.scalar(
        select(func.sum(func.length(FileRecord.extracted_text))).where(FileRecord.file_id.in_(ids))
    )
    if (length or 0) > settings.visual_max_source_chars:
        raise HTTPException(413, detail={"reason": "sources_too_large"})
    rows = list(
        await session.scalars(
            select(FileRecord)
            .options(undefer(FileRecord.extracted_text))
            .where(FileRecord.file_id.in_(ids))
        )
    )
    by_id = {r.file_id: r for r in rows}
    if not ids or len(rows) != len(ids):
        raise HTTPException(422, detail={"reason": "sources_missing"})
    sources = [
        {
            "id": id_,
            "name": by_id[id_].display_name or by_id[id_].original_name,
            "hash": source_hash(by_id[id_]),
            "text": by_id[id_].extracted_text,
            "raw_key": by_id[id_].raw_key,
        }
        for id_ in ids
    ]
    if sum(len(s["text"] or "") for s in sources) > settings.visual_max_source_chars:
        raise HTTPException(413, detail={"reason": "sources_too_large"})
    return {
        "title": case.title if case else sources[0]["name"],
        "sources": sources,
        "prompt_version": PROMPT_VERSION,
        "image": {
            "model": settings.visual_image_model,
            "size": settings.visual_image_size,
            "quality": settings.visual_image_quality,
        },
    }


async def check_sources(
    session: AsyncSession,
    artifact: ArtifactRecord,
    sources: list[dict],
    caller: str,
    *,
    require_current: bool = False,
    verify_content: bool = True,
) -> bool:
    """Current ACL applies to historical outputs too. Content changes mark stale."""
    from llm_wiki.api.v1.artifacts import _check_document_access

    await _check_document_access(session, artifact.document_id, caller)
    case = await session.get(CaseRecord, artifact.document_id)
    ids = [s["id"] for s in sources]
    current = set(case.doc_ids or []) if case else {artifact.document_id}
    stale = not set(ids) <= current
    query = select(FileRecord).where(FileRecord.file_id.in_(ids))
    if verify_content:
        query = query.options(undefer(FileRecord.extracted_text))
    by_id = {row.file_id: row for row in await session.scalars(query)}
    for source in sources:
        row = by_id.get(source["id"])
        if row is None or (row.sensitive and row.owner != caller):
            raise HTTPException(404, detail="Источник не найден")
        stale = (
            stale
            or row.status != "DONE"
            or (verify_content and bool(source.get("hash")) and source_hash(row) != source["hash"])
        )
    if require_current and stale:
        raise ValueError("sources_changed")
    return stale


async def start(
    session: AsyncSession,
    document_id: str,
    ids: list[str],
    language: str,
    caller: str,
    request_key: str | None,
    resume: bool = False,
) -> dict:
    if not settings.visual_presentations_enabled:
        raise HTTPException(503, detail={"reason": "visual_disabled"})
    if settings.llm_provider != "openai":
        raise HTTPException(503, detail={"reason": "image_provider_unavailable"})
    if language not in LANGUAGES:
        raise HTTPException(422, detail={"reason": "invalid_language"})
    if request_key is not None and (
        not isinstance(request_key, str) or not request_key.strip() or len(request_key) > 200
    ):
        raise HTTPException(422, detail={"reason": "invalid_request_key"})
    if not isinstance(resume, bool):
        raise HTTPException(422, detail={"reason": "invalid_resume"})
    snap = await snapshot(session, document_id, ids)
    # All replicas share this short admission lock. No provider call under it.
    await session.execute(text("SELECT pg_advisory_xact_lock(hashtext('visual:admission'))"))
    artifact = await session.scalar(
        select(ArtifactRecord)
        .where(
            ArtifactRecord.document_id == document_id, ArtifactRecord.kind == "presentation_visual"
        )
        .with_for_update()
    )
    key = digest([caller, document_id, request_key or uuid.uuid4().hex])
    prior = await session.scalar(select(VisualJob).where(VisualJob.request_key == key))
    if prior:
        if prior.language != language or prior.snapshot != snap:
            raise HTTPException(409, detail={"reason": "idempotency_conflict"})
        return {
            "artifact_id": prior.artifact_id,
            "kind": "presentation_visual",
            "status": prior.status,
            "generation_id": prior.id,
        }
    if artifact and artifact.status == "pending":
        job = await session.get(VisualJob, (artifact.generation_context or {}).get("id", ""))
        if job and job.status == "pending":
            if job.language != language or job.snapshot != snap:
                raise HTTPException(409, detail={"reason": "generation_pending"})
            return {
                "artifact_id": artifact.artifact_id,
                "kind": artifact.kind,
                "status": "pending",
                "generation_id": job.id,
            }
    pending = list(await session.scalars(select(VisualJob).where(VisualJob.status == "pending")))
    if any(j.requested_by == caller for j in pending):
        raise HTTPException(
            429, detail={"reason": "user_generation_limit"}, headers={"Retry-After": "15"}
        )
    if len(pending) >= settings.visual_max_queued:
        raise HTTPException(429, detail={"reason": "queue_full"}, headers={"Retry-After": "30"})
    old_job = (
        await session.get(VisualJob, (artifact.generation_context or {}).get("id", ""))
        if artifact
        else None
    )
    if old_job and await session.scalar(
        select(VisualUnit.job_id)
        .where(
            VisualUnit.job_id == old_job.id,
            VisualUnit.status == "running",
            VisualUnit.lease_until > now(),
        )
        .limit(1)
    ):
        raise HTTPException(409, detail={"reason": "retry_wait"}, headers={"Retry-After": "15"})
    if artifact is None:
        artifact = ArtifactRecord(
            artifact_id=uuid.uuid4().hex,
            document_id=document_id,
            kind="presentation_visual",
            versions=[],
        )
        session.add(artifact)
        await session.flush()
    job = VisualJob(
        id=uuid.uuid4().hex,
        artifact_id=artifact.artifact_id,
        language=language,
        requested_by=caller,
        request_key=key,
        snapshot=snap,
        deadline=now() + timedelta(seconds=settings.visual_job_timeout_s),
    )
    session.add(job)
    await session.flush()
    # Explicit retry reuses only completed units from exactly the same input.
    if (
        resume
        and old_job
        and old_job.status == "failed"
        and old_job.snapshot == snap
        and old_job.language == language
    ):
        previous = list(
            await session.scalars(select(VisualUnit).where(VisualUnit.job_id == old_job.id))
        )
        plan_unit = next((u for u in previous if u.index == 0 and u.status == "done"), None)
        job.plan = old_job.plan or (plan_unit.output if plan_unit else None)
        for unit in previous:
            # A late export may have finished without publishing a revision.
            # Rebuild the inexpensive files to execute atomic publication; all
            # already-paid plan/image units remain done and are never rerun.
            reuse = unit.status == "done" and unit.index != 9
            session.add(
                VisualUnit(
                    job_id=job.id,
                    index=unit.index,
                    status="done" if reuse else "queued",
                    output=unit.output if reuse else None,
                )
            )
        present = {u.index for u in previous}
        if job.plan:
            for n in range(1, len(job.plan["slides"]) + 1):
                if n not in present:
                    session.add(VisualUnit(job_id=job.id, index=n))
            complete = {u.index for u in previous if u.status == "done"}
            if set(range(1, len(job.plan["slides"]) + 1)) <= complete and 9 not in present:
                session.add(VisualUnit(job_id=job.id, index=9))
    else:
        session.add(VisualUnit(job_id=job.id, index=0, status="queued"))
    artifact.status, artifact.error = "pending", None
    artifact.requested_by, artifact.started_at, artifact.finished_at = caller, None, None
    artifact.generation_context = {"id": job.id, "language": language, "source_doc_ids": ids}
    await (
        session.commit()
    )  # Durable queued units are dispatched by beat, even if the broker is down now.
    from llm_wiki.observability import bind_entities

    bind_entities(artifact_id=artifact.artifact_id, document_id=document_id, generation_id=job.id)
    return {
        "artifact_id": artifact.artifact_id,
        "kind": artifact.kind,
        "status": "pending",
        "generation_id": job.id,
    }


async def progress(session: AsyncSession, record: ArtifactRecord) -> dict | None:
    if record.kind != "presentation_visual":
        return None
    job = await session.get(VisualJob, (record.generation_context or {}).get("id", ""))
    if not job:
        return None
    units = list(await session.scalars(select(VisualUnit).where(VisualUnit.job_id == job.id)))
    done = sum(1 for u in units if 1 <= u.index <= 8 and u.status == "done")
    active = next((u for u in units if u.status == "running"), None)
    stage = "planning" if not job.plan else "rendering"
    if any(u.index == 9 for u in units):
        stage = "exporting"
    if not active and not job.plan:
        stage = "queued"
    if job.status != "pending":
        stage = job.status
    return {
        "id": job.id,
        "status": job.status,
        "stage": stage,
        "language": job.language,
        "slides_done": done,
        "slides_total": len(job.plan["slides"]) if job.plan else 0,
        "error": job.error,
        "deadline": job.deadline.isoformat(),
    }


async def add_revision(
    session: AsyncSession, record: ArtifactRecord, language: str, content: dict, sources: list[dict]
) -> int:
    # Caller holds the artifact row lock, so revision numbers are monotonic.
    maximum = await session.scalar(
        select(func.max(ArtifactRevision.revision)).where(
            ArtifactRevision.artifact_id == record.artifact_id,
            ArtifactRevision.language == language,
        )
    )
    number = (maximum or 0) + 1
    session.add(
        ArtifactRevision(
            artifact_id=record.artifact_id,
            language=language,
            revision=number,
            content=content,
            sources=[{k: s[k] for k in ("id", "hash") if k in s} for s in sources],
            requested_by=record.requested_by,
        )
    )
    await session.flush()
    await session.execute(
        delete(ArtifactRevision).where(
            ArtifactRevision.artifact_id == record.artifact_id,
            ArtifactRevision.language == language,
            ArtifactRevision.revision <= number - 10,
        )
    )
    return number


async def store_editable_revision(
    session: AsyncSession,
    record: ArtifactRecord,
    language: str,
    content: dict,
    ids: list[str] | None,
) -> int:
    """The normal worker holds the artifact lock at publication."""

    async def refs(source_ids):
        rows = await session.scalars(
            select(FileRecord)
            .options(undefer(FileRecord.extracted_text))
            .where(FileRecord.file_id.in_(source_ids))
        )
        by_id = {r.file_id: r for r in rows}
        return [
            {"id": id_, **({"hash": source_hash(by_id[id_])} if id_ in by_id else {})}
            for id_ in source_ids
        ]

    case = await session.get(CaseRecord, record.document_id)
    fallback = list(case.doc_ids or []) if case else [record.document_id]
    for version in record.versions or []:
        lang = version.get("language")
        if (
            lang
            and version.get("content")
            and not await session.scalar(
                select(ArtifactRevision.revision)
                .where(
                    ArtifactRevision.artifact_id == record.artifact_id,
                    ArtifactRevision.language == lang,
                )
                .limit(1)
            )
        ):
            await add_revision(
                session,
                record,
                lang,
                version["content"],
                await refs(version.get("source_doc_ids") or fallback),
            )
    return await add_revision(
        session, record, language, content, await refs(ids if ids is not None else fallback)
    )
