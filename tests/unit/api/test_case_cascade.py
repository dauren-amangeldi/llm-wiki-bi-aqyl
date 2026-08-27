"""BUG-02 (UAT S1): удаление кейса должно удалять всё, что он принёс в базу.

Репро ревизии: удалённый кейс — первый результат поиска, счётчик материалов
не уменьшается. Контракт после фикса: DELETE /cases/{id} атомарно (в одной
транзакции) сносит осиротевшие файлы, их вики-страницы и эмбеддинги, артефакты,
твин-сессии и чат; S3-объекты уходят фоновой задачей. Файл, входящий в другой
кейс, НЕ сиротеет.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import patch

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llm_wiki.api.deps import get_db
from llm_wiki.main import app
from llm_wiki.storage.metadata import (
    ArtifactRecord,
    CaseRecord,
    ChatRecord,
    ChunkEmbedding,
    FileRecord,
    TwinMessage,
    TwinSession,
)
from llm_wiki.storage.wiki_fts import upsert_wiki_fts

_DIM = 1536


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


def _chunk(cid: str, slug: str, file_id: str) -> ChunkEmbedding:
    return ChunkEmbedding(
        id=cid, slug=slug, file_id=file_id, document="txt", embedding=[0.0] * _DIM
    )


async def _seed_world(db: AsyncSession) -> None:
    """Кейс A (f-own — только его; f-shared — общий с кейсом B) + все хвосты."""
    db.add(
        FileRecord(
            file_id="f-own", original_name="own.pdf", status="DONE",
            created_pages=["page-own"], raw_key="app/raw/2026/08/f-own.pdf",
        )
    )
    db.add(
        FileRecord(
            file_id="f-shared", original_name="shared.pdf", status="DONE",
            created_pages=["page-shared"],
        )
    )
    db.add(CaseRecord(id="case-a", title="A", doc_ids=["f-own", "f-shared"]))
    db.add(CaseRecord(id="case-b", title="B", doc_ids=["f-shared"]))
    await upsert_wiki_fts(db, slug="page-own", title="Own", body="секретный текст")
    await upsert_wiki_fts(db, slug="page-shared", title="Shared", body="общий текст")
    db.add(_chunk("page-own#0000", "page-own", "f-own"))
    db.add(_chunk("page-shared#0000", "page-shared", "f-shared"))
    db.add(
        ArtifactRecord(
            artifact_id="art-case", document_id="case-a", kind="report",
            versions=[], status="ready",
        )
    )
    db.add(
        ArtifactRecord(
            artifact_id="art-own-doc", document_id="f-own", kind="test",
            versions=[], status="ready",
        )
    )
    db.add(
        TwinSession(id="tw-1", case_id="case-a", persona_ids=["altman"], created_by="demo")
    )
    db.add(TwinMessage(id="twm-1", session_id="tw-1", role="user", seq=1, content={}))
    db.add(
        ChatRecord(
            user_key="demo", scope_type="case", scope_id="case-a",
            role="user", text="вопрос",
        )
    )
    await db.commit()


async def test_cascade_removes_everything_of_the_case(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    await _seed_world(db_session)

    with patch("llm_wiki.orchestrator.tasks.purge_case_objects") as purge_mock:
        resp = await client.delete("/api/v1/cases/case-a")
    assert resp.status_code == 200

    db_session.expire_all()
    assert await db_session.get(CaseRecord, "case-a") is None
    assert await db_session.get(FileRecord, "f-own") is None
    assert await db_session.get(ArtifactRecord, "art-case") is None
    assert await db_session.get(ArtifactRecord, "art-own-doc") is None
    assert await db_session.get(TwinSession, "tw-1") is None
    assert await db_session.get(TwinMessage, "twm-1") is None
    chats = (
        await db_session.scalars(
            select(ChatRecord).where(ChatRecord.scope_id == "case-a")
        )
    ).all()
    assert chats == []
    page = await db_session.execute(
        text("SELECT slug FROM wiki_fts WHERE slug = 'page-own'")
    )
    assert page.first() is None
    chunk = await db_session.get(ChunkEmbedding, "page-own#0000")
    assert chunk is None
    # S3-объект уходит в фоновую задачу с ключом файла.
    purge_mock.delay.assert_called_once_with(["app/raw/2026/08/f-own.pdf"])


async def test_shared_file_survives(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    await _seed_world(db_session)
    with patch("llm_wiki.orchestrator.tasks.purge_case_objects"):
        assert (await client.delete("/api/v1/cases/case-a")).status_code == 200

    db_session.expire_all()
    assert await db_session.get(FileRecord, "f-shared") is not None
    assert await db_session.get(ChunkEmbedding, "page-shared#0000") is not None
    page = await db_session.execute(
        text("SELECT slug FROM wiki_fts WHERE slug = 'page-shared'")
    )
    assert page.first() is not None
    assert await db_session.get(CaseRecord, "case-b") is not None


async def test_repeat_delete_is_a_clean_404(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    await _seed_world(db_session)
    with patch("llm_wiki.orchestrator.tasks.purge_case_objects"):
        assert (await client.delete("/api/v1/cases/case-a")).status_code == 200
        assert (await client.delete("/api/v1/cases/case-a")).status_code == 404


async def test_search_stops_finding_deleted_content(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """Сквозная проверка репро ревизии: после удаления кейса keyword-поиск
    больше не находит его страницу."""
    from llm_wiki.storage.wiki_fts import keyword_search

    await _seed_world(db_session)
    before = await keyword_search(db_session, "секретный")
    assert any("page-own" in str(hit) for hit in before)

    with patch("llm_wiki.orchestrator.tasks.purge_case_objects"):
        await client.delete("/api/v1/cases/case-a")

    after = await keyword_search(db_session, "секретный")
    assert not any("page-own" in str(hit) for hit in after)
