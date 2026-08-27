"""BUG-11: метрики в настройках — реальные цифры, контракт фронта."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from datetime import datetime, timedelta, timezone

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llm_wiki.api.deps import get_db
from llm_wiki.main import app
from llm_wiki.storage.metadata import ArtifactRecord, CaseRecord, ChatRecord, FileRecord


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
    ) as c:
        yield c
    app.dependency_overrides.clear()


async def test_metrics_shape_and_counts(client: AsyncClient, db_session: AsyncSession) -> None:
    now = datetime.now(timezone.utc)
    db_session.add(FileRecord(file_id="m-f1", original_name="a.pdf", status="DONE"))
    db_session.add(FileRecord(file_id="m-f2", original_name="b.pdf", status="PROCESSING"))
    db_session.add(CaseRecord(id="m-c1", title="C1", doc_ids=["m-f1"], tags=["Финансы", "Качество"]))
    db_session.add(CaseRecord(id="m-c2", title="C2", doc_ids=[], tags=["Финансы"]))
    db_session.add(
        ArtifactRecord(artifact_id="m-a1", document_id="m-c1", kind="report",
                       versions=[], status="ready", created_at=now)
    )
    db_session.add(
        ChatRecord(user_key="u", scope_type="case", scope_id="m-c1", role="user",
                   text="вопрос", created_at=now)
    )
    db_session.add(
        ChatRecord(user_key="u", scope_type="case", scope_id="m-c1", role="assistant",
                   text="ответ", created_at=now)
    )
    await db_session.commit()

    data = (await client.get("/api/v1/metrics")).json()
    assert data["stats"] == {
        "total_docs": 2, "total_cases": 2, "studied_cases": 1,
        "artifacts_generated": 1, "questions_asked": 1,
    }
    assert len(data["activity"]) == 7
    today = now.date().isoformat()
    today_row = next(r for r in data["activity"] if r["date"] == today)
    assert today_row["questions"] == 1 and today_row["generations"] == 1
    assert {"name": "Финансы", "count": 2} in data["tags_distribution"]


async def test_metrics_empty_db(client: AsyncClient, db_session: AsyncSession) -> None:
    data = (await client.get("/api/v1/metrics")).json()
    assert data["stats"]["total_docs"] == 0
    assert len(data["activity"]) == 7
    assert data["tags_distribution"] == []
