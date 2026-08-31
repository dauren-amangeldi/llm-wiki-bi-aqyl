"""Fail-fast генерации артефактов (QA-сценарий «Пересоздать по пустому кейсу»).

Контракт: (пере)генерация по заведомо пустому источнику отклоняется 422 в
момент запроса — ДО создания pending-строки и постановки в очередь. Прошлый
готовый артефакт при этом не трогается вовсе (остаётся ready со старым
содержимым), событий об ошибке в ленту не пишется — юзер уже увидел причину
мгновенным тостом.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llm_wiki.api.deps import get_db
from llm_wiki.main import app
from llm_wiki.storage.metadata import (
    ArtifactRecord,
    CaseRecord,
    FileRecord,
    NotificationRecord,
)

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


@pytest.mark.asyncio
async def test_regenerate_on_empty_case_422_keeps_old_artifact(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """Юзер-сценарий: удалил материалы → «Пересоздать» → 422 сразу, старый
    артефакт остаётся «Готово» со старым содержимым, ленте — ничего."""
    db_session.add(CaseRecord(id="case-empty", title="Пустой", doc_ids=[], owner=USER))
    db_session.add(ArtifactRecord(
        artifact_id="art-old", document_id="case-empty", kind="test",
        status="ready", versions=[{"language": "ru", "content": {"questions": []}}],
    ))
    await db_session.commit()

    resp = await client.post(
        "/api/v1/studio/generate",
        json={"kind": "test", "document_id": "case-empty", "language": "ru"},
    )
    assert resp.status_code == 422
    assert "нет материалов" in resp.json()["detail"].lower()

    db_session.expire_all()
    art = await db_session.get(ArtifactRecord, "art-old")
    assert art is not None
    assert art.status == "ready"          # НЕ pending и НЕ failed
    assert art.versions                    # старое содержимое цело
    # Никаких событий об ошибке: генерация даже не начиналась.
    assert (await db_session.scalars(select(NotificationRecord))).all() == []


@pytest.mark.asyncio
async def test_cards_generate_on_empty_case_422_without_rows(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """Синхронный путь (карточки): тоже 422 сразу, pending-строка не создаётся."""
    db_session.add(CaseRecord(id="case-e2", title="Пустой 2", doc_ids=[], owner=USER))
    await db_session.commit()

    resp = await client.post(
        "/api/v1/cards/generate",
        json={"document_id": "case-e2", "languages": ["ru"]},
    )
    assert resp.status_code == 422
    assert "нет материалов" in resp.json()["detail"].lower()

    assert (await db_session.scalars(select(ArtifactRecord))).all() == []
    assert (await db_session.scalars(select(NotificationRecord))).all() == []


@pytest.mark.asyncio
async def test_generate_while_materials_processing_422(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """Материалы есть, но вики-страниц ещё нет (обрабатываются) — 422 с
    понятной причиной, а не минутное фоновое падение."""
    db_session.add(FileRecord(file_id="f-proc", original_name="p.pdf",
                              status="SEARCHED", created_pages=[]))
    db_session.add(CaseRecord(id="case-proc", title="В работе",
                              doc_ids=["f-proc"], owner=USER))
    await db_session.commit()

    resp = await client.post(
        "/api/v1/studio/generate",
        json={"kind": "report", "document_id": "case-proc", "language": "ru"},
    )
    assert resp.status_code == 422
    assert "обрабатываются" in resp.json()["detail"].lower()
    assert (await db_session.scalars(select(ArtifactRecord))).all() == []


@pytest.mark.asyncio
async def test_valid_source_passes_the_guard(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """Кейс с обработанным материалом guard пропускает: создаётся pending
    (брокер в тестах недоступен → произойдёт inline-фолбэк, который упадёт на
    LLM — нам важно лишь, что 422 guard'а НЕ случился)."""
    db_session.add(FileRecord(file_id="f-ok", original_name="ok.pdf",
                              status="DONE", created_pages=["page-ok"]))
    db_session.add(CaseRecord(id="case-ok", title="Готовый",
                              doc_ids=["f-ok"], owner=USER))
    await db_session.commit()

    resp = await client.post(
        "/api/v1/studio/generate",
        json={"kind": "report", "document_id": "case-ok", "language": "ru"},
    )
    # Любой исход, кроме fail-fast 422 «нет материалов/без содержимого»:
    # генерация была ДОПУЩЕНА (детали её судьбы — вне этого теста).
    if resp.status_code == 422:
        detail = resp.json()["detail"].lower()
        assert "нет материалов" not in detail
        assert "без содержимого" not in detail