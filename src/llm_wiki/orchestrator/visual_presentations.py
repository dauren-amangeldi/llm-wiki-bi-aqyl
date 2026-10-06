"""Durable visual work: bounded dispatch, fenced units, atomic publication.

The DB queue is the outbox. Broker delivery is merely a wake-up: duplicated
messages cannot claim a running unit, and unpublished jobs are found by beat.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import timedelta

import openai
import structlog
from sqlalchemy import delete, select, text

from llm_wiki.agents.visual_presentation import (
    artwork_prompt,
    draw_artwork,
    export_deck,
    plan_deck,
    render_slide,
    thumbnail,
)
from llm_wiki.config import settings
from llm_wiki.storage.metadata import (
    ArtifactRecord,
    ArtifactRevision,
    NotificationRead,
    NotificationRecord,
    VisualJob,
    VisualUnit,
)
from llm_wiki.storage.object_store import get_object_store
from llm_wiki.storage.visual_presentations import add_revision, check_sources, now

logger = structlog.get_logger(__name__)
LOCK = text("SELECT pg_advisory_xact_lock(hashtext('visual:admission'))")


async def cleanup(factory) -> int:
    """Keep ten revisions/language; expire resumable attempts after 24h.

    Only immutable objects older than the grace period and unreachable from
    both revisions and still-resumable units are deleted. A new resume cannot
    race the snapshot expiration because admission uses the same DB lock.
    """
    cutoff = now() - timedelta(hours=24)
    references = set()
    async with factory() as session:
        await session.execute(LOCK)
        for job in await session.scalars(
            select(VisualJob).where(VisualJob.status != "pending", VisualJob.deadline < cutoff)
        ):
            await session.execute(delete(VisualUnit).where(VisualUnit.job_id == job.id))
            job.snapshot, job.plan = {"title": job.snapshot.get("title", ""), "expired": True}, None
        for content in await session.scalars(select(ArtifactRevision.content)):
            references.update(
                s["asset_key"] for s in content.get("slides", []) if s.get("asset_key")
            )
            references.update(
                s["thumbnail_key"] for s in content.get("slides", []) if s.get("thumbnail_key")
            )
            references.update(content.get("exports", {}).values())
        for output in await session.scalars(
            select(VisualUnit.output).where(VisualUnit.output.is_not(None))
        ):
            if output:
                if output.get("key"):
                    references.add(output["key"])
                if output.get("thumbnail_key"):
                    references.add(output["thumbnail_key"])
                references.update(output.get("exports", {}).values())
        await session.commit()
    store = get_object_store()
    removed = 0
    for obj in await asyncio.to_thread(store.list_objects, "artifacts/visual/"):
        if obj.last_modified < cutoff.timestamp() and obj.key not in references:
            await asyncio.to_thread(store.delete, obj.key)
            removed += 1
    return removed


async def _event(session, artifact, job, event: str, revision: int | None = None) -> None:
    from llm_wiki.storage.notifications import _event_statement

    stmt = _event_statement(
        section="artifacts",
        family="generation",
        event=event,
        entity_id=artifact.artifact_id,
        title=job.snapshot["title"],
        recipient=job.requested_by,
        actor=job.requested_by,
        detail=job.error,
        occurred_at=now(),
        occurrence_key=f"{job.id}:{event}",
        meta={
            "document_id": artifact.document_id,
            "kind": artifact.kind,
            "language": job.language,
            "revision": revision,
            "generation_id": job.id,
        },
    )
    notification_id = (
        await session.execute(stmt.returning(NotificationRecord.id))
    ).scalar_one_or_none()
    if notification_id:
        await session.execute(
            delete(NotificationRead).where(NotificationRead.notification_id == notification_id)
        )


async def fail(session, job: VisualJob, reason: str) -> None:
    from llm_wiki.observability import mark_outcome

    mark_outcome("failed")
    job.status, job.error = "failed", reason
    artifact = await session.get(ArtifactRecord, job.artifact_id)
    if artifact and (artifact.generation_context or {}).get("id") == job.id:
        artifact.status, artifact.error, artifact.finished_at = "failed", reason, now()
        await _event(session, artifact, job, "failed")


async def dispatch(factory) -> int:
    deliveries = []
    async with factory() as session:
        await session.execute(LOCK)
        jobs = list(
            await session.scalars(
                select(VisualJob)
                .where(VisualJob.status == "pending")
                .order_by(VisualJob.created_at)
            )
        )
        for job in jobs:
            artifact = await session.get(ArtifactRecord, job.artifact_id)
            if (
                not artifact
                or artifact.status != "pending"
                or (artifact.generation_context or {}).get("id") != job.id
            ):
                job.status, job.error = "failed", "cancelled"
            elif job.deadline <= now():
                await fail(session, job, "generation_timeout")
        units = list(
            await session.scalars(
                select(VisualUnit)
                .join(VisualJob)
                .where(VisualJob.status == "pending")
                .order_by(VisualJob.created_at, VisualUnit.index)
            )
        )
        by_job = {j.id: j for j in jobs if j.status == "pending"}
        for unit in units:
            if unit.lease_until and unit.lease_until < now():
                if unit.status == "running" and unit.job_id in by_job:
                    # Unknown provider outcome after worker loss: do not auto-pay again.
                    unit.status = "failed"
                    await fail(session, by_job.pop(unit.job_id), "worker_interrupted")
                elif unit.status == "dispatched":
                    unit.status, unit.token, unit.lease_until = "queued", None, None
        # Count running work even for failed jobs until its lease ends.
        active = list(
            await session.scalars(
                select(VisualUnit).where(
                    VisualUnit.status.in_(["running", "dispatched"]), VisualUnit.lease_until > now()
                )
            )
        )
        room = (
            max(0, settings.visual_max_active - len(active))
            if settings.visual_presentations_enabled
            else 0
        )
        for unit in units:
            if room <= 0:
                break
            if unit.status != "queued" or unit.job_id not in by_job:
                continue
            if unit.lease_until and unit.lease_until > now():  # retry backoff
                continue
            unit.status, unit.token = "dispatched", uuid.uuid4().hex
            unit.lease_until = now() + timedelta(seconds=60)
            deliveries.append((unit.job_id, unit.index, unit.token))
            room -= 1
        await session.commit()
    from llm_wiki.orchestrator.tasks import visual_unit

    for delivery in deliveries:
        try:
            visual_unit.apply_async(args=delivery, expires=60)
        except Exception:
            # The delivery lease expires and beat republishes. No lost job.
            logger.warning("visual_publish_delayed", job_id=delivery[0])
    return len(deliveries)


async def _produce(
    index: int,
    job_id: str,
    language: str,
    snapshot: dict,
    plan: dict | None,
    outputs: dict[int, dict],
    token: str,
) -> dict:
    store = get_object_store()
    prefix = f"artifacts/visual/{job_id}/{token}"
    if index == 0:
        from llm_wiki.orchestrator.pipeline import _load_raw_text

        sources = []
        for source in snapshot["sources"]:
            body = source["text"]
            if body is None:
                body = await asyncio.to_thread(_load_raw_text, source["id"], source["raw_key"])
            if not body or not body.strip():
                raise ValueError("sources_empty")
            sources.append({"id": source["id"], "name": source["name"], "text": body})
        if sum(len(s["text"]) for s in sources) > settings.visual_max_source_chars:
            raise ValueError("sources_too_large")
        return await plan_deck(
            {"title": snapshot["title"], "sources": sources, "image": snapshot["image"]},
            language,
            job_id,
        )
    if plan is None:
        raise ValueError("deck_invalid")
    if 1 <= index <= 8:
        slide = plan["slides"][index - 1]
        artwork = await draw_artwork(artwork_prompt(plan, slide), job_id, snapshot["image"])
        size = tuple(map(int, snapshot["image"]["size"].split("x")))
        data = await asyncio.to_thread(
            render_slide, slide, artwork, index, len(plan["slides"]), language, size
        )
        key = f"{prefix}/{index:02d}.png"
        await asyncio.to_thread(store.put_bytes, key, data)
        thumb_key = f"{prefix}/{index:02d}-thumb.png"
        thumb = await asyncio.to_thread(thumbnail, data)
        await asyncio.to_thread(store.put_bytes, thumb_key, thumb)
        return {"key": key, "thumbnail_key": thumb_key, "size": len(data) + len(thumb)}
    if sum(outputs[n]["size"] for n in range(1, len(plan["slides"]) + 1)) > 64_000_000:
        raise ValueError("deck_too_large")
    images = [
        await asyncio.to_thread(store.get_bytes, outputs[n]["key"])
        for n in range(1, len(plan["slides"]) + 1)
    ]
    exports = {}
    for fmt in ("pdf", "pptx"):
        key = f"{prefix}/deck.{fmt}"
        data = await asyncio.to_thread(export_deck, images, plan["slides"], fmt)
        await asyncio.to_thread(store.put_bytes, key, data)
        exports[fmt] = key
    return {"exports": exports}


async def work(factory, job_id: str, index: int, token: str) -> None:
    from llm_wiki.observability import bind_entities

    bind_entities(generation_id=job_id)
    async with factory() as session:
        await session.execute(LOCK)
        job = await session.get(VisualJob, job_id)
        unit = await session.get(VisualUnit, (job_id, index))
        if (
            not job
            or job.status != "pending"
            or not unit
            or unit.status != "dispatched"
            or unit.token != token
        ):
            return
        artifact = await session.get(ArtifactRecord, job.artifact_id)
        if artifact:
            bind_entities(artifact_id=artifact.artifact_id, document_id=artifact.document_id)
        if (
            not artifact
            or artifact.status != "pending"
            or (artifact.generation_context or {}).get("id") != job_id
        ):
            return
        if not settings.visual_presentations_enabled:
            unit.status = "failed"
            await fail(session, job, "visual_disabled")
            await session.commit()
            return
        try:
            await check_sources(
                session, artifact, job.snapshot["sources"], job.requested_by, require_current=True
            )
            if job.deadline <= now():
                raise ValueError("generation_timeout")
        except Exception as exc:
            unit.status = "failed"
            await fail(
                session,
                job,
                "generation_timeout" if str(exc) == "generation_timeout" else "sources_changed",
            )
            await session.commit()
            return
        unit.status, unit.lease_until = "running", now() + timedelta(seconds=400)
        unit.attempts += 1
        artifact.started_at = artifact.started_at or now()
        snapshot, plan, language = job.snapshot, job.plan, job.language
        outputs = {
            u.index: u.output
            for u in await session.scalars(
                select(VisualUnit).where(VisualUnit.job_id == job_id, VisualUnit.status == "done")
            )
        }
        await session.commit()
    result, reason, retry = None, None, False
    try:
        result = await asyncio.wait_for(
            _produce(index, job_id, language, snapshot, plan, outputs, token),
            settings.visual_unit_timeout_s,
        )
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        code = getattr(exc, "code", None)
        retry = (status == 429 and code != "insufficient_quota") or (
            status is not None and status >= 500
        )
        safe_reasons = {
            "sources_empty",
            "sources_too_large",
            "deck_invalid",
            "image_missing",
            "image_dimensions_invalid",
            "slide_text_overflow",
            "slide_sources_overflow",
            "deck_incomplete",
            "image_provider_unavailable",
            "image_capacity",
            "image_daily_limit",
            "image_limiter_unavailable",
            "image_too_large",
            "deck_too_large",
        }
        reason = (
            str(exc)
            if str(exc) in safe_reasons
            else "deck_invalid"
            if isinstance(exc, ValueError)
            else "provider_unavailable"
        )
        if reason == "image_capacity":
            retry = True
        if isinstance(exc, (TimeoutError, openai.APITimeoutError, openai.APIConnectionError)):
            reason = "provider_outcome_unknown"  # User decides whether to pay for another attempt.
        logger.warning(
            "visual_unit_failed",
            job_id=job_id,
            index=index,
            error_type=type(exc).__name__,
            reason=reason,
        )
    async with factory() as session:
        await session.execute(LOCK)
        job = await session.get(VisualJob, job_id)
        unit = await session.get(VisualUnit, (job_id, index))
        if not job or not unit or unit.token != token or unit.status != "running":
            return
        if job.status != "pending":
            unit.status, unit.output = ("done", result) if result else ("failed", None)
            await session.commit()
            return
        artifact = await session.get(ArtifactRecord, job.artifact_id, with_for_update=True)
        if (
            not artifact
            or artifact.status != "pending"
            or (artifact.generation_context or {}).get("id") != job.id
        ):
            return
        if reason:
            if (
                retry
                and (unit.attempts < 2 or reason == "image_capacity")
                and job.deadline > now() + timedelta(seconds=60)
            ):
                unit.status, unit.lease_until = "queued", now() + timedelta(seconds=20)
            else:
                unit.status = "failed"
                await fail(session, job, reason)
            await session.commit()
            return
        try:
            await check_sources(
                session, artifact, job.snapshot["sources"], job.requested_by, require_current=True
            )
            if job.deadline <= now():
                raise ValueError("generation_timeout")
        except Exception as exc:
            unit.status, unit.output = "done", result
            await fail(
                session,
                job,
                "generation_timeout" if str(exc) == "generation_timeout" else "sources_changed",
            )
            await session.commit()
            return
        unit.status, unit.output = "done", result
        if index == 0:
            job.plan = result
            for n in range(1, len(result["slides"]) + 1):
                session.add(VisualUnit(job_id=job.id, index=n))
        elif index < 9:
            await session.flush()
            slide_units = list(
                await session.scalars(
                    select(VisualUnit).where(
                        VisualUnit.job_id == job_id, VisualUnit.index.between(1, 8)
                    )
                )
            )
            if all(u.status == "done" for u in slide_units) and not await session.get(
                VisualUnit, (job_id, 9)
            ):
                session.add(VisualUnit(job_id=job_id, index=9))
        else:
            units = {
                u.index: u
                for u in await session.scalars(
                    select(VisualUnit).where(VisualUnit.job_id == job_id)
                )
            }
            content = {
                "schema_version": 1,
                "mode": "visual",
                "title": job.plan["title"],
                "style": {"name": "BI visual", "style_block": job.plan["style_block"]},
                "warnings": job.plan["warnings"],
                "createdAt": now().isoformat(),
                "slides": [
                    {
                        **s,
                        "asset_key": units[n + 1].output["key"],
                        "thumbnail_key": units[n + 1].output.get("thumbnail_key"),
                    }
                    for n, s in enumerate(job.plan["slides"])
                ],
                "exports": result["exports"],
            }
            revision = await add_revision(
                session, artifact, language, content, job.snapshot["sources"]
            )
            version = {
                "language": language,
                "revision": revision,
                "content": content,
                "source_doc_ids": [s["id"] for s in job.snapshot["sources"]],
            }
            artifact.versions = [
                v for v in (artifact.versions or []) if v.get("language") != language
            ] + [version]
            artifact.status, artifact.error, artifact.finished_at = "ready", None, now()
            job.status = "ready"
            await _event(session, artifact, job, "done", revision)
        await session.commit()
