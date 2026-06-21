"""Tests for skills storage (LW-N11)."""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from llm_wiki.storage.metadata import (
    Base,
    get_skill,
    list_skills,
    seed_skills,
    skill_role_to_slug,
    skill_slug_to_role,
    skills_count,
    update_skill,
)


@pytest.fixture
async def db_session() -> AsyncSession:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)
    async with factory() as session:
        yield session
    await engine.dispose()


async def test_seed_skills_inserts_all_roles(db_session: AsyncSession) -> None:
    inserted = await seed_skills(db_session)
    assert inserted == 10
    assert await skills_count(db_session) == 10

    roles = {s.role for s in await list_skills(db_session)}
    assert roles == {
        "advisor", "expert", "library",
        "employee", "finance", "gd", "hr", "legal", "pm", "pto",
    }


async def test_seed_skills_is_idempotent(db_session: AsyncSession) -> None:
    assert await seed_skills(db_session) == 10
    assert await seed_skills(db_session) == 0


async def test_skill_slug_mapping() -> None:
    assert skill_role_to_slug("advisor") == "modes/advisor"
    assert skill_role_to_slug("pm") == "positions/pm"
    assert skill_slug_to_role("modes/expert") == "expert"
    assert skill_slug_to_role("positions/hr") == "hr"


async def test_update_skill_system_prompt(db_session: AsyncSession) -> None:
    await seed_skills(db_session)
    updated = await update_skill(
        db_session,
        "pm",
        system_prompt="Custom PM prompt for testing.",
    )
    assert updated is not None
    assert updated.system_prompt == "Custom PM prompt for testing."

    reloaded = await get_skill(db_session, "pm")
    assert reloaded is not None
    assert reloaded.system_prompt == "Custom PM prompt for testing."
