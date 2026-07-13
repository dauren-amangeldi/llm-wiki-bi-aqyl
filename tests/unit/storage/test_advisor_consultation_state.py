"""Unit tests for AdvisorSession state/outcome transitions."""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llm_wiki.storage.metadata import (
    create_advisor_session,
    get_advisor_session,
    set_advisor_session_outcome,
    set_advisor_session_state,
)


@pytest.fixture
def session_factory(db_engine):
    return async_sessionmaker(bind=db_engine, expire_on_commit=False, autoflush=False)


async def test_new_session_defaults_to_discovery_state(session_factory) -> None:
    async with session_factory() as s:
        row = await create_advisor_session(s, user_key="alice@bi.group", title="Стоит ли строить в KZ?")
    assert row.state == "discovery"
    assert row.outcome is None


async def test_set_advisor_session_state_persists(session_factory) -> None:
    async with session_factory() as s:
        created = await create_advisor_session(s, user_key="alice@bi.group", title="Q")

    async with session_factory() as s:
        await set_advisor_session_state(s, created.id, "context_review")

    async with session_factory() as s:
        row = await get_advisor_session(s, created.id)
    assert row is not None
    assert row.state == "context_review"


async def test_set_advisor_session_outcome_persists(session_factory) -> None:
    async with session_factory() as s:
        created = await create_advisor_session(s, user_key="alice@bi.group", title="Q")

    async with session_factory() as s:
        await set_advisor_session_outcome(s, created.id, "decided")

    async with session_factory() as s:
        row = await get_advisor_session(s, created.id)
    assert row is not None
    assert row.outcome == "decided"
