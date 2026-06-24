"""Unit tests for wiki FTS5 keyword search (LW-N4 / LW-N5)."""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from llm_wiki.storage.metadata import Base
from llm_wiki.storage.wiki_fts import (
    _prepare_fts_query,
    ensure_wiki_fts_table,
    keyword_search,
    rebuild_wiki_fts_from_disk,
    upsert_wiki_fts,
)


@pytest.fixture
async def db_session(tmp_path):
    """Async session with FTS5 table ready."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/fts.db")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await ensure_wiki_fts_table(conn)
    factory = async_sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)
    async with factory() as session:
        yield session
    await engine.dispose()


async def test_keyword_search_finds_body_word(db_session: AsyncSession) -> None:
    """A word present in the page body returns slug and snippet."""
    await upsert_wiki_fts(
        db_session,
        slug="transformers",
        title="Transformers",
        body="# Transformers\n\nThe Adam optimizer is used for training large models.",
    )

    hits = await keyword_search(db_session, "optimizer", limit=5)

    assert len(hits) == 1
    assert hits[0].slug == "transformers"
    assert hits[0].title == "transformers"
    assert "optimizer" in hits[0].snippet.lower() or "Adam" in hits[0].snippet


async def test_keyword_search_short_query_returns_empty(db_session: AsyncSession) -> None:
    """Queries shorter than 3 chars are rejected upstream; FTS helper handles empty."""
    await upsert_wiki_fts(
        db_session,
        slug="test",
        title="Test",
        body="Some content here for testing purposes.",
    )
    assert await keyword_search(db_session, "ab", limit=5) == []


async def test_upsert_replaces_existing_slug(db_session: AsyncSession) -> None:
    """Re-indexing a slug replaces the old body (delete+insert)."""
    await upsert_wiki_fts(db_session, "page-a", "Old", "Content about apples only.")
    await upsert_wiki_fts(db_session, "page-a", "New", "Content about bananas instead.")

    apple_hits = await keyword_search(db_session, "apples", limit=5)
    banana_hits = await keyword_search(db_session, "bananas", limit=5)

    assert apple_hits == []
    assert len(banana_hits) == 1
    assert banana_hits[0].title == "new"


async def test_rebuild_from_disk(tmp_path, db_session: AsyncSession) -> None:
    """rebuild_wiki_fts_from_disk indexes all markdown files."""
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()
    (wiki_dir / "alpha.md").write_text(
        "# Alpha\n\nUnique keyword xyzzy appears here.",
        encoding="utf-8",
    )
    (wiki_dir / "beta.md").write_text(
        "# Beta\n\nAnother unique keyword plugh.",
        encoding="utf-8",
    )

    count = await rebuild_wiki_fts_from_disk(db_session, wiki_dir)
    assert count == 2

    hits = await keyword_search(db_session, "xyzzy", limit=5)
    assert len(hits) == 1
    assert hits[0].slug == "alpha"


def test_prepare_fts_query_uses_or_not_and() -> None:
    """Question-style queries use OR so partial token overlap still matches."""
    q = _prepare_fts_query("Как настроить отчёт по продажам?")
    assert " OR " in q
    assert " AND " not in q
    # Long tokens are stemmed to a prefix (Russian morphology): настроить -> настро*
    assert "настро" in q
    assert "отчет" in q  # ё folded
    assert '"как"*' not in q  # stop-word dropped


async def test_keyword_search_question_phrase(db_session: AsyncSession) -> None:
    """Natural-language question finds a page without requiring every word."""
    await upsert_wiki_fts(
        db_session,
        slug="sales-report",
        title="Отчёт по продажам",
        body="Настройка отчёта по продажам в дашборде.",
    )

    hits = await keyword_search(db_session, "как настроить отчет по продажам", limit=5)

    assert len(hits) >= 1
    assert hits[0].slug == "sales-report"


async def test_keyword_search_yo_e_fold(db_session: AsyncSession) -> None:
    """ё/е normalization works in both directions (query ↔ indexed body)."""
    await upsert_wiki_fts(
        db_session,
        slug="report-e",
        title="Report",
        body="Документ про отчет без буквы ё.",
    )

    hits_yo = await keyword_search(db_session, "отчёт", limit=5)
    assert len(hits_yo) >= 1
    assert hits_yo[0].slug == "report-e"

    from llm_wiki.storage.wiki_fts import delete_wiki_fts

    await delete_wiki_fts(db_session, "report-e")

    await upsert_wiki_fts(
        db_session,
        slug="report-yo",
        title="Report",
        body="Документ про отчёт с буквой ё.",
    )

    hits_e = await keyword_search(db_session, "отчет", limit=5)
    assert len(hits_e) >= 1
    assert hits_e[0].slug == "report-yo"
