"""GET /api/v1/ops/generations — мониторинг генераций для Grafana (В1/В2)."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import datetime, timedelta, timezone

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llm_wiki.api.deps import get_db
from llm_wiki.main import app
from llm_wiki.storage.metadata import ArtifactRecord, FileRecord


@pytest_asyncio.fixture
async def client(db_engine) -> AsyncGenerator[AsyncClient, None]:
    session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
        bind=db_engine, expire_on_commit=False, autoflush=False
    )

    async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db

    async with AsyncClient(
        transport=ASGITransport(app=app),  # type: ignore[arg-type]
        base_url="http://test",
    ) as c:
        yield c

    app.dependency_overrides.clear()


async def _seed(db_session: AsyncSession) -> None:
    now = datetime.now(timezone.utc)
    db_session.add(
        ArtifactRecord(
            artifact_id="a-run", document_id="doc-1", kind="report",
            versions=[], status="pending",
            requested_by="alice@bi.group", started_at=now - timedelta(seconds=30),
        )
    )
    db_session.add(
        ArtifactRecord(
            artifact_id="a-queued", document_id="doc-1", kind="test",
            versions=[], status="pending", requested_by="bob@bi.group",
        )
    )
    db_session.add(
        ArtifactRecord(
            artifact_id="a-fail", document_id="doc-2", kind="presentation",
            versions=[], status="failed", error="boom",
            requested_by="alice@bi.group",
            started_at=now - timedelta(minutes=2), finished_at=now - timedelta(minutes=1),
        )
    )
    db_session.add(
        FileRecord(
            file_id="f-active", original_name="big.pdf", status="PROCESSING",
            owner="carol@bi.group",
        )
    )
    await db_session.commit()


async def test_ops_generations_reports_status_author_and_queues(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    await _seed(db_session)

    resp = await client.get("/api/v1/ops/generations")
    assert resp.status_code == 200
    data = resp.json()

    # Очереди присутствуют всегда (None — если брокер недоступен в тестах).
    assert set(data["queues"].keys()) == {"ingest", "artifacts", "light"}

    by_id = {it["id"]: it for it in data["items"]}
    assert by_id["a-run"]["status"] == "running"
    assert by_id["a-run"]["requested_by"] == "alice@bi.group"
    assert by_id["a-run"]["duration_s"] is not None
    assert by_id["a-queued"]["status"] == "queued"
    assert by_id["a-fail"]["status"] == "failed"
    assert by_id["a-fail"]["error"] == "boom"
    assert by_id["f-active"]["status"] == "running"
    assert by_id["f-active"]["requested_by"] == "carol@bi.group"
    assert by_id["f-active"]["what"] == "big.pdf"

    # Активные — раньше завершённых.
    statuses = [it["status"] for it in data["items"]]
    assert statuses.index("running") < statuses.index("failed")


async def test_ops_generations_requires_token_when_configured(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    from llm_wiki.config import settings

    old = settings.ops_token
    settings.ops_token = "sekret"
    try:
        assert (await client.get("/api/v1/ops/generations")).status_code == 403
        ok = await client.get(
            "/api/v1/ops/generations", headers={"X-Ops-Token": "sekret"}
        )
        assert ok.status_code == 200
    finally:
        settings.ops_token = old


async def test_purge_marks_stuck_generations_failed(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """POST /ops/queues/purge: pending-артефакты и активные файлы → failed с
    понятной причиной (даже когда брокер недоступен, как в тестах)."""
    db_session.add(
        ArtifactRecord(
            artifact_id="a-stuck", document_id="doc-9", kind="report",
            versions=[], status="pending",
        )
    )
    db_session.add(
        FileRecord(file_id="f-stuck", original_name="stuck.pdf", status="PROCESSING")
    )
    await db_session.commit()

    resp = await client.post("/api/v1/ops/queues/purge", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert data["artifacts_marked_failed"] >= 1
    assert data["files_marked_failed"] >= 1
    assert set(data["purged"].keys()) == {"ingest", "artifacts", "light", "celery"}

    db_session.expire_all()
    art = await db_session.get(ArtifactRecord, "a-stuck")
    fr = await db_session.get(FileRecord, "f-stuck")
    assert art is not None and art.status == "failed" and "Отменено" in (art.error or "")
    assert fr is not None and fr.status == "FAILED"
