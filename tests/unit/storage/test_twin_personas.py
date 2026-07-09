"""Unit tests for Twins persona/preset seeding (storage/metadata.py)."""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.storage.metadata import (
    get_twin_persona,
    list_twin_presets,
    seed_twin_personas,
    twin_personas_count,
)


@pytest.mark.asyncio
async def test_seed_twin_personas_inserts_eleven_rows(db_session: AsyncSession) -> None:
    inserted = await seed_twin_personas(db_session)
    assert inserted == 11
    assert await twin_personas_count(db_session) == 11


@pytest.mark.asyncio
async def test_seed_twin_personas_second_pass_inserts_nothing_new(db_session: AsyncSession) -> None:
    await seed_twin_personas(db_session)
    second_pass = await seed_twin_personas(db_session)
    assert second_pass == 0
    assert await twin_personas_count(db_session) == 11


@pytest.mark.asyncio
async def test_seed_twin_personas_upserts_a_row_edited_after_first_seed(db_session: AsyncSession) -> None:
    """Editing a persona's .md file and restarting must update the DB row.

    Simulated here by mutating the row directly (standing in for "the file
    changed since the last seed") and confirming the next seed overwrites it —
    this is what makes the personas file-editable instead of insert-once.
    """
    await seed_twin_personas(db_session)
    musk = await get_twin_persona(db_session, "musk")
    assert musk is not None
    musk.lens = "stale value predating an edit"
    await db_session.commit()

    await seed_twin_personas(db_session)

    refreshed = await get_twin_persona(db_session, "musk")
    assert refreshed is not None
    assert refreshed.lens != "stale value predating an edit"
    assert refreshed.lens == "Радикальная себестоимость, автоматизация"


@pytest.mark.asyncio
async def test_musk_persona_has_tech_track_and_domain_weights(db_session: AsyncSession) -> None:
    await seed_twin_personas(db_session)
    musk = await get_twin_persona(db_session, "musk")
    assert musk is not None
    assert musk.track == "tech"
    assert musk.pinned == 1
    assert musk.domain_weights["tech"] == 0.9


@pytest.mark.asyncio
async def test_presets_reference_seeded_persona_ids(db_session: AsyncSession) -> None:
    await seed_twin_personas(db_session)
    presets = await list_twin_presets(db_session)
    names = {p.name for p in presets}
    assert "Технотрансформация" in names
    tech_preset = next(p for p in presets if p.name == "Технотрансформация")
    assert tech_preset.persona_ids == ["musk", "huang", "nadella"]
