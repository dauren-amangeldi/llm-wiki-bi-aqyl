"""Б1: POST /files/{id}/retry — переобработка упавшего файла с карточки.

Контракт: только терминальный FAILED, только владелец (или файл без владельца);
статус сбрасывается в RECEIVED, error чистится, poison-cap обнуляется, задача
уходит в очередь. state_history не трогаем — пайплайн продолжает с места сбоя.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llm_wiki.api.deps import get_db
from llm_wiki.main import app
from llm_wiki.storage.metadata import FileRecord

USER = "demo@bi.group"


@pytest_asyncio.fixture
async def client(db_engine) -> AsyncGenerator[AsyncClient, None]:
    factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
        bind=db_engine, expire_on_commit=False, autoflush=False
    )

    async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with factory() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db
    async with AsyncClient(
        transport=ASGITransport(app=app),  # type: ignore[arg-type]
        base_url="http://test",
        headers={"X-User-Email": USER},
    ) as c:
        yield c
    app.dependency_overrides.clear()


def _delay_mock() -> MagicMock:
    task = MagicMock()
    task.id = "task-1"
    delay = MagicMock(return_value=task)
    return delay


@pytest.mark.asyncio
async def test_retry_resets_failed_file_and_enqueues(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    db_session.add(FileRecord(
        file_id="f-fail", original_name="a.pdf", status="FAILED",
        error="упс", ingest_attempts=5, owner=USER,
        state_history=[{"state": "STORED", "at": "2026-08-24T00:00:00Z"}],
    ))
    await db_session.commit()

    with patch("llm_wiki.api.routes.process_file_task") as task_mock:
        task_mock.delay = _delay_mock()
        resp = await client.post("/api/v1/files/f-fail/retry")

    assert resp.status_code == 200
    assert resp.json()["status"] == "queued"
    task_mock.delay.assert_called_once_with("f-fail")

    db_session.expire_all()
    fr = await db_session.get(FileRecord, "f-fail")
    assert fr is not None
    assert fr.status == "RECEIVED"
    assert fr.error is None
    assert fr.ingest_attempts == 0
    # Резюме пайплайна: история стадий сохранена.
    assert fr.state_history == [{"state": "STORED", "at": "2026-08-24T00:00:00Z"}]


@pytest.mark.asyncio
async def test_retry_rejects_non_failed_and_foreign(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    db_session.add(FileRecord(file_id="f-done", original_name="b.pdf", status="DONE"))
    db_session.add(FileRecord(
        file_id="f-foreign", original_name="c.pdf", status="FAILED",
        owner="someone@bi.group",
    ))
    await db_session.commit()

    with patch("llm_wiki.api.routes.process_file_task") as task_mock:
        task_mock.delay = _delay_mock()
        assert (await client.post("/api/v1/files/f-done/retry")).status_code == 409
        assert (await client.post("/api/v1/files/f-foreign/retry")).status_code == 403
        assert (await client.post("/api/v1/files/missing/retry")).status_code == 404
        task_mock.delay.assert_not_called()
