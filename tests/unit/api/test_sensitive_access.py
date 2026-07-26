"""Endpoint-level access control: sensitive docs/cases are owner-scoped."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llm_wiki.api.deps import get_db
from llm_wiki.main import app
from llm_wiki.storage.metadata import CaseRecord, FileRecord

_ALICE = {"X-User-Email": "alice@bi.group"}
_BOB = {"X-User-Email": "bob@bi.group"}


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


async def test_documents_list_hides_others_sensitive(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    db_session.add(FileRecord(file_id="d-pub", original_name="pub.md", status="DONE"))
    db_session.add(
        FileRecord(
            file_id="d-sec", original_name="sec.md", status="DONE",
            sensitive=True, owner="alice@bi.group",
        )
    )
    await db_session.commit()

    bob_ids = {m["document_id"] for m in (await client.get("/api/v1/documents", headers=_BOB)).json()}
    assert "d-pub" in bob_ids
    assert "d-sec" not in bob_ids  # ← must not leak

    alice_ids = {m["document_id"] for m in (await client.get("/api/v1/documents", headers=_ALICE)).json()}
    assert {"d-pub", "d-sec"} <= alice_ids


async def test_get_document_access_control(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    db_session.add(
        FileRecord(
            file_id="d-sec2", original_name="s.md", status="DONE",
            sensitive=True, owner="alice@bi.group",
        )
    )
    await db_session.commit()

    assert (await client.get("/api/v1/documents/d-sec2", headers=_BOB)).status_code == 404
    ra = await client.get("/api/v1/documents/d-sec2", headers=_ALICE)
    assert ra.status_code == 200
    assert ra.json()["sensitive"] is True


async def test_cases_list_hides_others_private(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    db_session.add(CaseRecord(id="c-pub", title="Pub", doc_ids=[], sensitive=False))
    db_session.add(
        CaseRecord(id="c-priv", title="Priv", doc_ids=[], sensitive=True, owner="alice@bi.group")
    )
    await db_session.commit()

    bob_ids = {c["id"] for c in (await client.get("/api/v1/cases", headers=_BOB)).json()}
    assert "c-pub" in bob_ids
    assert "c-priv" not in bob_ids  # ← must not leak
