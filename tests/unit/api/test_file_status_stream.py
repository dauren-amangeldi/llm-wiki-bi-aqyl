"""Ingestion status: failure-reason persistence + the SSE status stream."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llm_wiki.api.deps import get_db
from llm_wiki.main import app
from llm_wiki.storage.metadata import FileRecord, get_file_record, update_file_status


@pytest_asyncio.fixture
async def client(db_engine) -> AsyncGenerator[AsyncClient, None]:
    factory = async_sessionmaker(bind=db_engine, expire_on_commit=False, autoflush=False)

    async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with factory() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db
    async with AsyncClient(
        transport=ASGITransport(app=app),  # type: ignore[arg-type]
        base_url="http://test",
    ) as ac:
        yield ac
    app.dependency_overrides.clear()


def _first_frame(sse_text: str) -> dict:
    """Parse the first ``data:`` SSE frame from a stream body."""
    for line in sse_text.splitlines():
        if line.startswith("data:"):
            return json.loads(line[5:].strip())
    raise AssertionError(f"no SSE data frame in: {sse_text!r}")


# ---------------------------------------------------------------------------
# Failure reason is persisted (visible without logs)
# ---------------------------------------------------------------------------


async def test_update_file_status_persists_error(db_session: AsyncSession) -> None:
    db_session.add(FileRecord(file_id="fse-1", original_name="z.md", status="STORED"))
    await db_session.commit()

    await update_file_status(db_session, "fse-1", "FAILED", error="kaboom")
    db_session.expire_all()  # force a fresh read (mirrors a separate connection)

    rec = await get_file_record(db_session, "fse-1")
    assert rec is not None
    assert rec.status == "FAILED"
    assert rec.error == "kaboom"


# ---------------------------------------------------------------------------
# SSE status stream — terminal states emit and close immediately
# ---------------------------------------------------------------------------


async def test_status_stream_done_emits_terminal(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    db_session.add(
        FileRecord(
            file_id="fse-done",
            original_name="a.md",
            status="DONE",
            created_pages=["slug-a"],
        )
    )
    await db_session.commit()

    resp = await client.get("/api/v1/files/fse-done/status/stream")
    assert resp.status_code == 200
    frame = _first_frame(resp.text)
    assert frame["done"] is True
    assert frame["ok"] is True
    assert "slug-a" in frame["pages"]


async def test_status_stream_failed_carries_reason(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    db_session.add(
        FileRecord(file_id="fse-fail", original_name="b.md", status="FAILED", error="boom")
    )
    await db_session.commit()

    resp = await client.get("/api/v1/files/fse-fail/status/stream")
    assert resp.status_code == 200
    frame = _first_frame(resp.text)
    assert frame["done"] is True
    assert frame["ok"] is False
    assert frame["error"] == "boom"


async def test_status_stream_missing_file(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/files/does-not-exist/status/stream")
    assert resp.status_code == 200  # stream opens, then reports not_found + done
    frame = _first_frame(resp.text)
    assert frame.get("error") == "not_found"
    assert frame["done"] is True
