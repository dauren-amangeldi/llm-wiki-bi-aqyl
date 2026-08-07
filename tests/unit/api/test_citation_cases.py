"""Task 2: answer citations carry their source case (the "source case" chip)."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llm_wiki.api.deps import get_db
from llm_wiki.main import app
from llm_wiki.storage.metadata import CaseRecord, append_chat_message, case_for_file


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


async def test_case_for_file_resolves_owning_case(db_session: AsyncSession) -> None:
    """A source's file_id resolves to the case whose doc_ids contain it."""
    db_session.add(CaseRecord(id="c-a", title="Кейс A", doc_ids=["f1", "f2"]))
    db_session.add(CaseRecord(id="c-b", title="Кейс B", doc_ids=["f3"]))
    await db_session.commit()

    assert await case_for_file(db_session, "f2") == ("c-a", "Кейс A")
    assert await case_for_file(db_session, "f3") == ("c-b", "Кейс B")
    assert await case_for_file(db_session, "unlinked") is None


async def test_case_chat_history_carries_source_case_chip(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """A persisted answer keeps its source-case chip on reload."""
    await append_chat_message(
        db_session,
        user_key="anon",
        scope_type="case",
        scope_id="c-x",
        role="assistant",
        text_body="Ответ [[private-f1]].",
        citations=["private-f1"],
        citation_cases={"private-f1": {"id": "c-x", "title": "Кейс X"}},
    )

    turns = (await client.get("/api/v1/cases/c-x/chat")).json()
    assert len(turns) == 1
    cite = turns[0]["citations"][0]
    assert cite["anchor"] == "private-f1"
    assert cite["case_id"] == "c-x"
    assert cite["case_title"] == "Кейс X"
