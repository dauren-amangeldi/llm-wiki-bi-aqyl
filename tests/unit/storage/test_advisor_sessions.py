"""Unit tests for AdvisorSession storage helpers."""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llm_wiki.storage.metadata import (
    create_advisor_session,
    get_advisor_session,
    list_advisor_sessions,
    touch_advisor_session,
)


@pytest.fixture
def session_factory(db_engine):
    return async_sessionmaker(bind=db_engine, expire_on_commit=False, autoflush=False)


async def test_create_advisor_session_persists_title_and_owner(session_factory) -> None:
    async with session_factory() as s:
        row = await create_advisor_session(s, user_key="alice@bi.group", title="Стоит ли строить в KZ?")
    assert row.id.startswith("advisor-session-")
    assert row.user_key == "alice@bi.group"
    assert row.title == "Стоит ли строить в KZ?"


async def test_list_advisor_sessions_scoped_to_user_newest_first(session_factory) -> None:
    async with session_factory() as s:
        await create_advisor_session(s, user_key="alice@bi.group", title="First")
        await create_advisor_session(s, user_key="bob@bi.group", title="Not Alice's")
        await create_advisor_session(s, user_key="alice@bi.group", title="Second")

    async with session_factory() as s:
        rows = await list_advisor_sessions(s, user_key="alice@bi.group")
    assert [r.title for r in rows] == ["Second", "First"]


async def test_get_advisor_session_returns_none_for_unknown_id(session_factory) -> None:
    async with session_factory() as s:
        row = await get_advisor_session(s, "does-not-exist")
    assert row is None


async def test_touch_advisor_session_bumps_updated_at_and_reorders_list(session_factory) -> None:
    async with session_factory() as s:
        first = await create_advisor_session(s, user_key="alice@bi.group", title="First")
        await create_advisor_session(s, user_key="alice@bi.group", title="Second")

    async with session_factory() as s:
        await touch_advisor_session(s, first.id)

    async with session_factory() as s:
        rows = await list_advisor_sessions(s, user_key="alice@bi.group")
    assert rows[0].title == "First"  # touched, so now newest
