"""Unit tests for case-embedding refresh and similarity search."""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.config import settings
from llm_wiki.storage.metadata import (
    CaseRecord,
    ChunkEmbedding,
    find_similar_cases,
    refresh_case_embedding,
)

_DIM = settings.embedding_dimensions


def _vec(value: float, dim: int = _DIM) -> list[float]:
    """A constant vector — cheap way to build controllable cosine distances."""
    return [value] * dim


async def _add_chunk(session: AsyncSession, chunk_id: str, file_id: str, vector: list[float]) -> None:
    session.add(
        ChunkEmbedding(
            id=chunk_id,
            slug=f"slug-{file_id}",
            file_id=file_id,
            document="body",
            embedding=vector,
        )
    )


async def test_refresh_case_embedding_returns_false_without_docs(db_session: AsyncSession) -> None:
    db_session.add(CaseRecord(id="case-empty", title="Empty", doc_ids=[]))
    await db_session.commit()

    assert await refresh_case_embedding(db_session, "case-empty") is False


async def test_refresh_case_embedding_returns_false_before_ingestion_finishes(
    db_session: AsyncSession,
) -> None:
    db_session.add(CaseRecord(id="case-pending", title="Pending", doc_ids=["doc-not-embedded-yet"]))
    await db_session.commit()

    assert await refresh_case_embedding(db_session, "case-pending") is False


async def test_refresh_case_embedding_averages_chunk_vectors(db_session: AsyncSession) -> None:
    db_session.add(CaseRecord(id="case-a", title="Case A", doc_ids=["doc-1", "doc-2"]))
    await db_session.commit()

    # 0.2 and 0.4 should average to ~0.3 per dimension.
    await _add_chunk(db_session, "doc-1#0000", "doc-1", _vec(0.2))
    await _add_chunk(db_session, "doc-2#0000", "doc-2", _vec(0.4))
    await db_session.commit()

    assert await refresh_case_embedding(db_session, "case-a") is True

    from llm_wiki.storage.metadata import CaseEmbedding

    stored = await db_session.get(CaseEmbedding, "case-a")
    assert stored is not None
    assert stored.embedding[0] == pytest.approx(0.3, abs=1e-4)


async def test_find_similar_cases_ranks_closest_first(db_session: AsyncSession) -> None:
    # Case A and B are near-identical vectors; case C is far away.
    db_session.add_all([
        CaseRecord(id="case-a", title="Северный квартал", doc_ids=["doc-a"]),
        CaseRecord(id="case-b", title="Южный квартал (похожий)", doc_ids=["doc-b"]),
        CaseRecord(id="case-c", title="Совсем другой кейс", doc_ids=["doc-c"]),
    ])
    await db_session.commit()

    await _add_chunk(db_session, "doc-a#0000", "doc-a", _vec(0.5))
    await _add_chunk(db_session, "doc-b#0000", "doc-b", _vec(0.51))
    await _add_chunk(db_session, "doc-c#0000", "doc-c", [0.5] * (_DIM // 2) + [-0.5] * (_DIM - _DIM // 2))
    await db_session.commit()

    for cid in ("case-a", "case-b", "case-c"):
        assert await refresh_case_embedding(db_session, cid) is True

    matches = await find_similar_cases(db_session, "case-a", limit=5)

    assert [m[0] for m in matches] == ["case-b", "case-c"]
    assert matches[0][2] > matches[1][2]  # similarity_pct: closer case ranks higher


async def test_find_similar_cases_empty_when_no_embedding(db_session: AsyncSession) -> None:
    db_session.add(CaseRecord(id="case-lonely", title="Lonely", doc_ids=[]))
    await db_session.commit()

    assert await find_similar_cases(db_session, "case-lonely") == []
