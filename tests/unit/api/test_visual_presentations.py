from __future__ import annotations

import asyncio
import io
from datetime import timedelta
from unittest.mock import Mock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from PIL import Image
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from llm_wiki.api.deps import get_db
from llm_wiki.config import settings
from llm_wiki.main import app
from llm_wiki.orchestrator.visual_presentations import dispatch, work
from llm_wiki.storage.metadata import (
    ArtifactRecord,
    ArtifactRevision,
    CaseRecord,
    FileRecord,
    NotificationRecord,
    VisualJob,
    VisualUnit,
)
from llm_wiki.storage.visual_presentations import now

USER = "demo@bi.group"
SLIDE = {
    "title": "Рост",
    "subtitle": "",
    "points": ["1 500 объектов"],
    "layout": "editorial",
    "image_brief": "Glass architecture",
    "source_refs": ["file-v"],
    "speaker_notes": "Заметки",
}
PLAN = {
    "title": "Проверка",
    "style_block": "White and blue",
    "warnings": [],
    "slides": [SLIDE, SLIDE],
}


@pytest_asyncio.fixture
async def setup(db_engine, monkeypatch):
    factory = async_sessionmaker(db_engine, expire_on_commit=False)
    monkeypatch.setattr(settings, "visual_presentations_enabled", True)
    monkeypatch.setattr(settings, "visual_max_active", 2)

    async def db():
        async with factory() as s:
            yield s

    app.dependency_overrides[get_db] = db
    async with factory() as s:
        s.add(
            FileRecord(
                file_id="file-v",
                original_name="source.txt",
                status="DONE",
                created_pages=["wiki-v"],
                extracted_text="1 500 объектов",
                owner=USER,
            )
        )
        s.add(CaseRecord(id="case-v", title="Проверка", doc_ids=["file-v"], owner=USER))
        await s.commit()
    from llm_wiki.orchestrator.tasks import visual_unit

    monkeypatch.setattr(visual_unit, "apply_async", Mock())
    import llm_wiki.orchestrator.visual_presentations as pipeline

    async def plan(*args):
        return PLAN

    async def draw(*args):
        buf = io.BytesIO()
        Image.new("RGB", (1536, 864), "white").save(buf, "PNG")
        return buf.getvalue()

    monkeypatch.setattr(pipeline, "plan_deck", plan)
    monkeypatch.setattr(pipeline, "draw_artwork", draw)
    async with AsyncClient(
        transport=ASGITransport(app), base_url="http://test", headers={"X-User-Email": USER}
    ) as client:
        yield client, factory
    app.dependency_overrides.clear()


async def start(client, **extra):
    return await client.post(
        "/api/v1/studio/generate",
        json={"kind": "presentation_visual", "document_id": "case-v", "language": "ru", **extra},
    )


async def advance(factory):
    await dispatch(factory)
    async with factory() as s:
        deliveries = [
            (u.job_id, u.index, u.token)
            for u in await s.scalars(select(VisualUnit).where(VisualUnit.status == "dispatched"))
        ]
    for delivery in deliveries:
        await work(factory, *delivery)


@pytest.mark.asyncio
async def test_visual_end_to_end_revision_export_and_private_assets(setup):
    client, factory = setup
    response = await start(client, request_key="click-1")
    assert response.status_code == 202, response.text
    aid = response.json()["artifact_id"]
    assert (await start(client, request_key="click-1")).json() == response.json()
    for _ in range(3):
        await advance(factory)
    detail = (await client.get(f"/api/v1/artifacts/{aid}?language=ru")).json()
    assert detail["status"] == "ready", detail
    assert detail["generation"]["slides_done"] == 2
    assert "exports" not in detail["versions"][0]["content"]
    image = detail["versions"][0]["content"]["slides"][0]
    assert "asset_key" not in image
    assert "thumbnail_key" not in image
    assert (await client.get(image["image"])).headers["content-type"] == "image/png"
    preview = await client.get(image["thumbnail"])
    assert preview.status_code == 200
    assert preview.headers["cache-control"] == "private, no-store"
    assert Image.open(io.BytesIO(preview.content)).size == (384, 216)
    assert detail["generation"]["stage"] == "ready"
    pdf = await client.get(f"/api/v1/artifacts/{aid}/export?format=pdf&language=ru&revision=1")
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")
    assert (await client.get(f"/api/v1/artifacts/{aid}?language=kk")).status_code == 404
    assert len((await client.get(f"/api/v1/artifacts/{aid}/revisions?language=ru")).json()) == 1
    async with factory() as s:
        notifications = list(await s.scalars(select(NotificationRecord)))
        assert len(notifications) == 1 and notifications[0].recipient == USER
        assert notifications[0].meta["revision"] == 1
        file = await s.get(FileRecord, "file-v")
        file.sensitive = True
        await s.commit()
    assert (
        await client.get(image["image"], headers={"X-User-Email": "other@bi.group"})
    ).status_code == 404
    assert (
        await client.get(image["thumbnail"], headers={"X-User-Email": "other@bi.group"})
    ).status_code == 404
    # Replayed idempotency key cannot buy another generation after success.
    assert (await start(client, request_key="click-1")).json()["status"] == "ready"


@pytest.mark.asyncio
async def test_failed_slide_resume_preserves_successes(setup, monkeypatch):
    client, factory = setup
    aid = (await start(client)).json()["artifact_id"]
    await advance(factory)  # plan
    import llm_wiki.orchestrator.visual_presentations as pipeline

    original = pipeline.draw_artwork
    calls = 0

    async def flaky(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("image_missing")
        return await original(*args)

    monkeypatch.setattr(pipeline, "draw_artwork", flaky)
    await advance(factory)
    assert (await client.get(f"/api/v1/artifacts/{aid}?language=ru")).json()["status"] == "failed"
    assert (await start(client, resume=True)).status_code == 202
    for _ in range(2):
        await advance(factory)
    assert calls == 3  # one completed image reused, not repurchased
    assert (await client.get(f"/api/v1/artifacts/{aid}?language=ru")).json()["status"] == "ready"


@pytest.mark.asyncio
async def test_worker_loss_and_duplicate_delivery_are_bounded(setup):
    client, factory = setup
    response = (await start(client)).json()
    await dispatch(factory)
    async with factory() as s:
        unit = await s.get(VisualUnit, (response["generation_id"], 0))
        token = unit.token
    await work(factory, response["generation_id"], 0, token)
    await work(factory, response["generation_id"], 0, token)
    async with factory() as s:
        assert len(list(await s.scalars(select(VisualUnit)))) == 3
        unit = await s.get(VisualUnit, (response["generation_id"], 1))
        unit.status, unit.lease_until = "running", now() - timedelta(seconds=1)
        await s.commit()
    await dispatch(factory)
    async with factory() as s:
        job = await s.get(VisualJob, response["generation_id"])
        assert job.status == "failed" and job.error == "worker_interrupted"


@pytest.mark.asyncio
async def test_sources_empty_changed_and_double_admission(setup):
    client, factory = setup
    assert (await start(client, source_doc_ids=[])).status_code == 422
    result = (await start(client)).json()
    assert (await start(client)).json()["generation_id"] == result["generation_id"]
    assert (await start(client, language="kk")).status_code == 409
    async with factory() as s:
        source = await s.get(FileRecord, "file-v")
        source.extracted_text = "Changed content"
        await s.commit()
    await advance(factory)
    async with factory() as s:
        job = await s.get(VisualJob, result["generation_id"])
        assert job.status == "failed" and job.error == "sources_changed"


@pytest.mark.asyncio
async def test_old_success_survives_failed_regeneration(setup, monkeypatch):
    client, factory = setup
    aid = (await start(client)).json()["artifact_id"]
    for _ in range(3):
        await advance(factory)
    import llm_wiki.orchestrator.visual_presentations as pipeline

    async def bad_plan(*args):
        raise ValueError("deck_invalid")

    monkeypatch.setattr(pipeline, "plan_deck", bad_plan)
    await start(client)
    await advance(factory)
    detail = (await client.get(f"/api/v1/artifacts/{aid}?language=ru")).json()
    assert detail["status"] == "failed" and detail["versions"][0]["revision"] == 1
    async with factory() as s:
        assert len(list(await s.scalars(select(ArtifactRevision)))) == 1


@pytest.mark.asyncio
async def test_dispatch_replicas_and_broker_failure_cannot_overbook(setup, monkeypatch):
    client, factory = setup
    result = (await start(client)).json()
    await advance(factory)
    from llm_wiki.orchestrator.tasks import visual_unit

    monkeypatch.setattr(
        visual_unit, "apply_async", Mock(side_effect=RuntimeError("broker unavailable"))
    )
    assert sum(await asyncio.gather(dispatch(factory), dispatch(factory))) == 2
    assert await dispatch(factory) == 0
    async with factory() as s:
        units = list(await s.scalars(select(VisualUnit).where(VisualUnit.status == "dispatched")))
        old_token = units[0].token
        for unit in units:
            unit.lease_until = now() - timedelta(seconds=1)
        await s.commit()
    assert await dispatch(factory) == 2
    await work(factory, result["generation_id"], 1, old_token)
    async with factory() as s:
        unit = await s.get(VisualUnit, (result["generation_id"], 1))
        assert unit.status == "dispatched" and unit.attempts == 0


@pytest.mark.asyncio
async def test_bad_inputs_and_cancelled_artifact_never_start_work(setup):
    client, factory = setup
    assert (await start(client, request_key=42)).status_code == 422
    assert (await start(client, resume="yes")).status_code == 422
    result = (await start(client)).json()
    async with factory() as s:
        row = await s.get(ArtifactRecord, result["artifact_id"])
        row.status = "failed"  # ops purge
        await s.commit()
    assert await dispatch(factory) == 0
    async with factory() as s:
        job = await s.get(VisualJob, result["generation_id"])
        assert job.status == "failed" and job.error == "cancelled"


@pytest.mark.asyncio
async def test_retry_waits_for_old_inflight_slides(setup):
    client, factory = setup
    result = (await start(client)).json()
    await advance(factory)
    async with factory() as s:
        row = await s.get(ArtifactRecord, result["artifact_id"])
        job = await s.get(VisualJob, result["generation_id"])
        unit = await s.get(VisualUnit, (job.id, 1))
        row.status = job.status = "failed"
        unit.status, unit.lease_until = "running", now() + timedelta(seconds=30)
        await s.commit()
    response = await start(client, resume=True)
    assert response.status_code == 409 and response.json()["detail"]["reason"] == "retry_wait"


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["plan", "slides", "export"])
async def test_resume_recovers_completed_result_at_deadline_boundary(setup, monkeypatch, boundary):
    """Late results must fan out or export on retry without another paid call."""
    from sqlalchemy import delete

    import llm_wiki.orchestrator.visual_presentations as pipeline

    client, factory = setup
    first = (await start(client)).json()
    await advance(factory)
    if boundary in {"slides", "export"}:
        await advance(factory)
    if boundary == "export":
        await advance(factory)
    async with factory() as session:
        job = await session.get(VisualJob, first["generation_id"])
        artifact = await session.get(ArtifactRecord, first["artifact_id"])
        # A completed provider result arrives after the job deadline. It is
        # saved, but no downstream unit is created by the failed attempt.
        job.status = artifact.status = "failed"
        if boundary == "plan":
            job.plan = None
            await session.execute(
                delete(VisualUnit).where(VisualUnit.job_id == job.id, VisualUnit.index > 0)
            )
        elif boundary == "slides":
            await session.execute(
                delete(VisualUnit).where(VisualUnit.job_id == job.id, VisualUnit.index == 9)
            )
        else:
            artifact.versions = []
            await session.execute(
                delete(ArtifactRevision).where(ArtifactRevision.artifact_id == artifact.artifact_id)
            )
        await session.commit()

    async def no_second_plan(*args):
        raise AssertionError("completed plan must be reused")

    async def no_second_image(*args):
        raise AssertionError("completed image must be reused")

    monkeypatch.setattr(pipeline, "plan_deck", no_second_plan)
    if boundary in {"slides", "export"}:
        monkeypatch.setattr(pipeline, "draw_artwork", no_second_image)
    response = await start(client, resume=True)
    assert response.status_code == 202, response.text
    assert response.json()["generation_id"] != first["generation_id"]
    for _ in range(2):
        await advance(factory)
    detail = (await client.get(f"/api/v1/artifacts/{first['artifact_id']}?language=ru")).json()
    assert detail["status"] == "ready", detail
    assert detail["generation"]["slides_done"] == 2


@pytest.mark.asyncio
async def test_editable_history_is_durable_and_language_does_not_fallback(setup):
    client, factory = setup
    from llm_wiki.storage.artifacts_store import upsert_artifact

    async with factory() as s:
        row = await upsert_artifact(
            s,
            document_id="case-v",
            kind="presentation",
            language="ru",
            content={
                "title": "First",
                "slides": [{"heading": "First", "bullets": ["Old content"]}],
            },
            source_doc_ids=["file-v"],
        )
        aid = row.artifact_id
        await upsert_artifact(
            s,
            document_id="case-v",
            kind="presentation",
            language="ru",
            content={
                "title": "Second",
                "slides": [{"heading": "Second", "bullets": ["New content"]}],
            },
            source_doc_ids=["file-v"],
        )
    old = (await client.get(f"/api/v1/artifacts/{aid}?language=ru&revision=1")).json()
    assert old["versions"][0]["content"]["title"] == "First"
    assert (await client.get(f"/api/v1/artifacts/{aid}?language=kk")).status_code == 404
    assert (
        await client.get(f"/api/v1/artifacts/{aid}/export?language=kk&format=pptx")
    ).status_code == 404
    assert len((await client.get(f"/api/v1/artifacts/{aid}/revisions?language=ru")).json()) == 2
    async with factory() as s:
        for n in range(10):
            await upsert_artifact(
                s,
                document_id="case-v",
                kind="presentation",
                language="ru",
                content={"slides": [{"heading": str(n)}]},
                source_doc_ids=["file-v"],
            )
    assert (await client.get(f"/api/v1/artifacts/{aid}?language=ru&revision=1")).status_code == 404
    assert len((await client.get(f"/api/v1/artifacts/{aid}/revisions?language=ru")).json()) == 10


@pytest.mark.asyncio
async def test_retention_keeps_referenced_assets_and_removes_deleted_case_objects(
    setup, monkeypatch
):
    import os

    from sqlalchemy import delete

    from llm_wiki.orchestrator.visual_presentations import cleanup
    from llm_wiki.storage.object_store import LocalObjectStore

    client, factory = setup
    import llm_wiki.orchestrator.visual_presentations as pipeline

    store = pipeline.get_object_store()
    assert isinstance(store, LocalObjectStore)
    result = (await start(client)).json()
    for _ in range(3):
        await advance(factory)
    objects = store.list_objects("artifacts/visual/")
    keys = [o.key for o in objects if result["generation_id"] in o.key]
    for key in keys:
        os.utime(store._path(key), (1, 1))
    async with factory() as s:
        job = await s.get(VisualJob, result["generation_id"])
        job.deadline = now() - timedelta(days=2)
        await s.commit()
    await cleanup(factory)
    assert all(store.exists(key) for key in keys)
    async with factory() as s:
        await s.execute(
            delete(ArtifactRecord).where(ArtifactRecord.artifact_id == result["artifact_id"])
        )
        await s.commit()
    await cleanup(factory)
    assert all(not store.exists(key) for key in keys)
