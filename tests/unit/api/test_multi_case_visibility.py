"""QA-фикс 1.3: материал может состоять в нескольких кейсах.

Правило видимости: файл приватен, ТОЛЬКО если все содержащие его кейсы
приватны. Публикация любого кейса открывает материал; «сделать приватным» не
прячет файл, пока тот входит в другой общий кейс.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llm_wiki.api.deps import get_db
from llm_wiki.main import app
from llm_wiki.storage.metadata import CaseRecord, FileRecord

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


async def _file_sensitive(db: AsyncSession, fid: str) -> bool:
    db.expire_all()
    fr = await db.get(FileRecord, fid)
    assert fr is not None
    return bool(fr.sensitive)


def _case_body(case: CaseRecord, *, sensitive: bool) -> dict:
    return {
        "title": case.title,
        "doc_ids": case.doc_ids,
        "sensitive": sensitive,
        "tags": [],
        "scope": "internal",
    }


@pytest.mark.asyncio
async def test_private_case_does_not_hide_file_shared_with_public_case(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """Файл в общем кейсе A и приватном B: сохранение B не прячет файл."""
    db_session.add(FileRecord(file_id="f-shared", original_name="s.pdf", status="DONE"))
    a = CaseRecord(id="case-pub", title="A", doc_ids=["f-shared"],
                   sensitive=False, owner=USER)
    b = CaseRecord(id="case-priv", title="B", doc_ids=["f-shared"],
                   sensitive=True, owner=USER)
    db_session.add_all([a, b])
    await db_session.commit()

    # Пересохраняем ПРИВАТНЫЙ кейс B — файл должен остаться публичным (кейс A).
    r = await client.put("/api/v1/cases/case-priv", json=_case_body(b, sensitive=True))
    assert r.status_code == 200
    assert await _file_sensitive(db_session, "f-shared") is False


@pytest.mark.asyncio
async def test_file_goes_private_only_when_all_cases_private(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """Последний общий кейс стал приватным → файл наконец приватный."""
    db_session.add(FileRecord(file_id="f-x", original_name="x.pdf", status="DONE"))
    a = CaseRecord(id="case-a", title="A", doc_ids=["f-x"], sensitive=False, owner=USER)
    b = CaseRecord(id="case-b", title="B", doc_ids=["f-x"], sensitive=True, owner=USER)
    db_session.add_all([a, b])
    await db_session.commit()

    # Делаем ПОСЛЕДНИЙ общий кейс A приватным → других общих нет → файл приватный.
    r = await client.put("/api/v1/cases/case-a", json=_case_body(a, sensitive=True))
    assert r.status_code == 200
    assert await _file_sensitive(db_session, "f-x") is True


@pytest.mark.asyncio
async def test_publishing_any_case_opens_the_file(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """Файл приватен (оба кейса приватны) → публикация одного кейса открывает."""
    db_session.add(FileRecord(file_id="f-y", original_name="y.pdf", status="DONE",
                              sensitive=True, owner=USER))
    a = CaseRecord(id="case-c", title="C", doc_ids=["f-y"], sensitive=True, owner=USER)
    b = CaseRecord(id="case-d", title="D", doc_ids=["f-y"], sensitive=True, owner=USER)
    db_session.add_all([a, b])
    await db_session.commit()

    r = await client.put("/api/v1/cases/case-c", json=_case_body(a, sensitive=False))
    assert r.status_code == 200
    assert await _file_sensitive(db_session, "f-y") is False
