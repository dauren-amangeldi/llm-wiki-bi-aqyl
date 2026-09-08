"""CRUD for generated studio artifacts (``ArtifactRecord``).

One row per (document_id, kind); regenerating replaces that language's version.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.storage.metadata import ArtifactRecord


async def get_artifact(session: AsyncSession, artifact_id: str) -> ArtifactRecord | None:
    return await session.get(ArtifactRecord, artifact_id)


async def list_artifacts(session: AsyncSession, document_id: str) -> list[ArtifactRecord]:
    rows = await session.scalars(
        select(ArtifactRecord).where(ArtifactRecord.document_id == document_id)
    )
    return list(rows)


async def find_by_kind(
    session: AsyncSession, document_id: str, kind: str
) -> ArtifactRecord | None:
    return await session.scalar(
        select(ArtifactRecord).where(
            ArtifactRecord.document_id == document_id,
            ArtifactRecord.kind == kind,
        )
    )


async def upsert_artifact(
    session: AsyncSession,
    *,
    document_id: str,
    kind: str,
    language: str,
    content: dict[str, Any],
    source_doc_ids: list[str] | None = None,
) -> ArtifactRecord:
    """Store (or replace) the ``language`` version of the (document, kind) artifact.

    Reassigns ``versions`` to a new list so SQLAlchemy tracks the JSON change.
    """
    record = await find_by_kind(session, document_id, kind)
    version: dict[str, Any] = {"language": language, "content": content}
    if source_doc_ids is not None:
        version["source_doc_ids"] = list(source_doc_ids)
    if record is None:
        record = ArtifactRecord(
            artifact_id=uuid.uuid4().hex,
            document_id=document_id,
            kind=kind,
            versions=[version],
            status="ready",
            finished_at=datetime.now(timezone.utc),
        )
        session.add(record)
        try:
            await session.commit()
        except IntegrityError:
            # Lost a concurrent-insert race on uq_artifacts_document_kind —
            # reuse the winner's row instead of failing the whole generation.
            await session.rollback()
            record = await find_by_kind(session, document_id, kind)
            if record is None:  # pragma: no cover — winner vanished mid-race
                raise
            others = [
                v
                for v in (record.versions or [])
                if isinstance(v, dict) and v.get("language") != language
            ]
            record.versions = [*others, version]
            record.status = "ready"
            record.finished_at = datetime.now(timezone.utc)
            await session.commit()
        return record
    others = [
        v
        for v in (record.versions or [])
        if isinstance(v, dict) and v.get("language") != language
    ]
    record.versions = [*others, version]
    record.status = "ready"
    record.finished_at = datetime.now(timezone.utc)
    await session.commit()
    return record


async def create_pending_artifact(
    session: AsyncSession, *, document_id: str, kind: str, requested_by: str | None = None,
    generation_context: dict[str, Any] | None = None
) -> ArtifactRecord:
    """Create (or reset to) a ``pending`` artifact for (document, kind).

    Used before enqueuing async generation: the endpoint returns this id right
    away and the client polls until ``status`` becomes ``ready``/``failed``. On
    regeneration the existing row is reused (old versions kept, so the reader
    still shows the previous content while the new one is generated).
    """
    record = await find_by_kind(session, document_id, kind)
    if record is None:
        record = ArtifactRecord(
            artifact_id=uuid.uuid4().hex,
            document_id=document_id,
            kind=kind,
            versions=[],
            status="pending",
            requested_by=requested_by,
            generation_context=generation_context,
            started_at=None,
            finished_at=None,
        )
        session.add(record)
        try:
            await session.commit()
        except IntegrityError:
            # Concurrent create for the same (document, kind): reuse the row.
            await session.rollback()
            record = await find_by_kind(session, document_id, kind)
            if record is None:  # pragma: no cover — winner vanished mid-race
                raise
            record.generation_context = generation_context
            record.status = "pending"
            record.error = None
            record.requested_by = requested_by
            record.started_at = None
            record.finished_at = None
            await session.commit()
        return record
    record.generation_context = generation_context
    record.status = "pending"
    record.error = None
    record.requested_by = requested_by
    record.started_at = None
    record.finished_at = None
    await session.commit()
    return record


async def mark_started(session: AsyncSession, artifact_id: str) -> None:
    """Stamp the moment a worker actually began generating (ops monitoring)."""
    record = await session.get(ArtifactRecord, artifact_id)
    if record is not None:
        record.started_at = datetime.now(timezone.utc)
        await session.commit()


async def mark_failed(
    session: AsyncSession, artifact_id: str, error: str, *, generation_id: str | None = None,
) -> bool:
    """Flag an artifact's generation as failed (keeps any prior versions)."""
    record = await session.get(ArtifactRecord, artifact_id, populate_existing=True, with_for_update=True)
    if record is not None:
        if generation_id is not None and (record.status != "pending" or (record.generation_context or {}).get("id") != generation_id):
            return False
        record.status = "failed"
        record.error = error[:500]
        record.finished_at = datetime.now(timezone.utc)
        await session.commit()
        return True
    return False


def serialize_detail(record: ArtifactRecord) -> dict[str, Any]:
    """Shape a record for GET /artifacts/{id} (versions = [{language, content}])."""
    versions = [
        {"language": v.get("language", ""), "content": v.get("content"),
         "source_doc_ids": v.get("source_doc_ids")}
        for v in (record.versions or [])
        if isinstance(v, dict)
    ]
    return {
        "artifact_id": record.artifact_id,
        "kind": record.kind,
        "status": record.status,
        "error": record.error,
        "versions": versions,
    }


def serialize_summary(record: ArtifactRecord, language: str = "ru") -> dict[str, Any]:
    """Shape a record for GET /artifacts (list)."""
    versions = [v for v in (record.versions or []) if isinstance(v, dict)]
    version = next((v for v in versions if v.get("language") == language), versions[0] if versions else {})
    return {
        "has_content": bool(versions),
        "source_doc_ids": version.get("source_doc_ids"),
        "artifact_id": record.artifact_id,
        "kind": record.kind,
        "status": record.status,
        # QA D5: плитка студии показывает failed-состояние с причиной в тултипе.
        "error": record.error,
        "created_at": record.created_at.isoformat() if record.created_at else "",
    }
