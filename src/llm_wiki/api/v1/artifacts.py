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

from fastapi import Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.agents.artifacts import ArtifactError, generate_content
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


async def _generate_and_store(
    session: AsyncSession, kind: str, document_id: str, language: str
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
    return record.artifact_id, content


@router.post("/studio/generate")
async def studio_generate(
    body: dict[str, Any],
    session: AsyncSession = Depends(get_db),
    _caller: str = Depends(get_user_key),
) -> dict[str, Any]:
    kind = str(body.get("kind") or "")
    document_id = str(body.get("document_id") or "")
    language = str(body.get("language") or "ru")
    if kind not in ("report", "test", "presentation"):
        raise HTTPException(status_code=400, detail=f"Unsupported kind for /studio/generate: {kind!r}")
    if not document_id:
        raise HTTPException(status_code=400, detail="document_id is required")
    try:
        artifact_id, content = await _generate_and_store(session, kind, document_id, language)
    except ArtifactError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"artifact_id": artifact_id, "kind": kind, "content": content}


@router.post("/cards/generate")
async def cards_generate(
    body: dict[str, Any],
    session: AsyncSession = Depends(get_db),
    _caller: str = Depends(get_user_key),
) -> dict[str, Any]:
    document_id = str(body.get("document_id") or "")
    langs = body.get("languages")
    language = str(langs[0]) if isinstance(langs, list) and langs else "ru"
    if not document_id:
        raise HTTPException(status_code=400, detail="document_id is required")
    try:
        artifact_id, _ = await _generate_and_store(session, "card", document_id, language)
    except ArtifactError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"artifact_id": artifact_id, "kind": "card"}


@router.post("/images/generate")
async def images_generate(
    body: dict[str, Any],
    session: AsyncSession = Depends(get_db),
    _caller: str = Depends(get_user_key),
) -> dict[str, Any]:
    document_id = str(body.get("document_id") or "")
    language = str(body.get("language") or "ru")
    if not document_id:
        raise HTTPException(status_code=400, detail="document_id is required")
    try:
        artifact_id, content = await _generate_and_store(session, "infographic", document_id, language)
    except ArtifactError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"artifact_id": artifact_id, "kind": "infographic", "svg": content.get("svg")}
