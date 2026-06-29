"""Unit tests for wiki full-text search (PostgreSQL tsvector, LW-N4)."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.storage.object_store import get_object_store, wiki_key
from llm_wiki.storage.wiki_fts import (
    delete_wiki_fts,
    keyword_search,
    rebuild_wiki_fts_from_store,
    upsert_wiki_fts,
)

# ``db_session`` comes from conftest (Postgres, wiki_fts table ready).


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
    # Postgres stores original casing (to_tsvector normalises for matching only).
    assert hits[0].title == "Transformers"
    assert "optimizer" in hits[0].snippet.lower()


async def test_keyword_search_empty_query_returns_empty(db_session: AsyncSession) -> None:
    """Blank queries return no hits without touching the DB."""
    await upsert_wiki_fts(db_session, slug="test", title="Test", body="Some content here.")
    assert await keyword_search(db_session, "   ", limit=5) == []


async def test_upsert_replaces_existing_slug(db_session: AsyncSession) -> None:
    """Re-indexing a slug replaces the old body."""
    await upsert_wiki_fts(db_session, "page-a", "Old", "Content about apples only.")
    await upsert_wiki_fts(db_session, "page-a", "New", "Content about bananas instead.")

    apple_hits = await keyword_search(db_session, "apples", limit=5)
    banana_hits = await keyword_search(db_session, "bananas", limit=5)

    assert apple_hits == []
    assert len(banana_hits) == 1
    assert banana_hits[0].title == "New"


async def test_rebuild_from_store(db_session: AsyncSession) -> None:
    """rebuild_wiki_fts_from_store indexes every wiki page in the object store."""
    store = get_object_store()
    store.put_text(wiki_key("alpha"), "# Alpha\n\nUnique keyword xyzzy appears here.")
    store.put_text(wiki_key("beta"), "# Beta\n\nAnother unique keyword plugh.")

    count = await rebuild_wiki_fts_from_store(db_session)
    assert count == 2

    hits = await keyword_search(db_session, "xyzzy", limit=5)
    assert len(hits) == 1
    assert hits[0].slug == "alpha"


async def test_keyword_search_russian_morphology(db_session: AsyncSession) -> None:
    """Russian stemmer matches inflected forms (ценности → ценностей)."""
    await upsert_wiki_fts(
        db_session,
        slug="oyo-values",
        title="Ошибки ценностей",
        body="Компания строила работу вокруг ценностей и стратегии роста.",
    )

    hits = await keyword_search(db_session, "ценности", limit=5)
    assert len(hits) == 1
    assert hits[0].slug == "oyo-values"


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


async def test_delete_removes_from_index(db_session: AsyncSession) -> None:
    """delete_wiki_fts drops a page from the index."""
    await upsert_wiki_fts(db_session, "doc", "Doc", "Документ про продажи.")
    assert len(await keyword_search(db_session, "продажи", limit=5)) == 1

    await delete_wiki_fts(db_session, "doc")
    assert await keyword_search(db_session, "продажи", limit=5) == []
