"""Tests for persona suggestion from similar cases (R2-2)."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.config import settings
from llm_wiki.storage.metadata import (
    CaseRecord,
    ChunkEmbedding,
    create_twin_session,
    refresh_case_embedding,
    suggest_twin_personas,
)

_DIM = settings.embedding_dimensions


async def _case_with_embedding(
    session: AsyncSession, case_id: str, title: str, value: float
) -> None:
    doc_id = f"doc-{case_id}"
    session.add(CaseRecord(id=case_id, title=title, doc_ids=[doc_id]))
    session.add(
        ChunkEmbedding(
            id=f"{doc_id}#0000",
            slug=f"slug-{doc_id}",
            file_id=doc_id,
            document="body",
            embedding=[value] * _DIM,
        )
    )
    await session.commit()
    assert await refresh_case_embedding(session, case_id) is True


async def test_suggests_personas_from_most_similar_case(db_session: AsyncSession) -> None:
    await _case_with_embedding(db_session, "case-new", "Новый кейс", 0.5)
    await _case_with_embedding(db_session, "case-old", "Старый похожий", 0.51)

    await create_twin_session(
        db_session, case_id="case-old",
        persona_ids=["musk", "zell", "hines"], created_by="u1",
    )

    suggestion = await suggest_twin_personas(db_session, "case-new")

    assert suggestion is not None
    assert suggestion["case_id"] == "case-old"
    assert suggestion["case_title"] == "Старый похожий"
    assert suggestion["persona_ids"] == ["musk", "zell", "hines"]
    assert suggestion["similarity_pct"] > 99


async def test_skips_similar_cases_without_councils(db_session: AsyncSession) -> None:
    await _case_with_embedding(db_session, "case-new", "Новый кейс", 0.5)
    await _case_with_embedding(db_session, "case-nearest", "Ближайший, но без совета", 0.51)
    await _case_with_embedding(db_session, "case-farther", "Дальше, но с советом", 0.6)

    await create_twin_session(
        db_session, case_id="case-farther",
        persona_ids=["altman"], created_by="u1",
    )

    suggestion = await suggest_twin_personas(db_session, "case-new")

    assert suggestion is not None
    assert suggestion["case_id"] == "case-farther"


async def test_returns_none_when_no_similar_case_has_council(db_session: AsyncSession) -> None:
    await _case_with_embedding(db_session, "case-new", "Новый кейс", 0.5)
    await _case_with_embedding(db_session, "case-other", "Другой", 0.51)

    assert await suggest_twin_personas(db_session, "case-new") is None


async def test_returns_none_for_case_without_embedding(db_session: AsyncSession) -> None:
    db_session.add(CaseRecord(id="case-bare", title="Без документов", doc_ids=[]))
    await db_session.commit()

    assert await suggest_twin_personas(db_session, "case-bare") is None
