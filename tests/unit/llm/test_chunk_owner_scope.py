"""Security core of sensitive files: chunk retrieval is owner-scoped.

A sensitive chunk must be returned ONLY to its owner — never to another user
or to an anonymous caller. The filter is applied in ChunkStore.query.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from sqlalchemy.orm import Session

from llm_wiki.config import settings
from llm_wiki.llm.chunk_store import ChunkStore
from llm_wiki.storage.metadata import ChunkEmbedding

_VEC = [0.1] * settings.embedding_dimensions


def _seed(engine) -> None:  # type: ignore[no-untyped-def]
    with Session(engine) as s:
        s.add(
            ChunkEmbedding(
                id="pub#0000", slug="public-page", title="Public", section="",
                chunk_idx=0, file_id="f-pub", sensitive=False, owner=None,
                document="public content", embedding=_VEC,
            )
        )
        s.add(
            ChunkEmbedding(
                id="priv#0000", slug="private-f-sec", title="Secret", section="",
                chunk_idx=0, file_id="f-sec", sensitive=True, owner="alice@bi.group",
                document="secret salary data", embedding=_VEC,
            )
        )
        s.commit()


def _store(engine) -> ChunkStore:  # type: ignore[no-untyped-def]
    llm = MagicMock()
    llm.embed.return_value = [_VEC]
    return ChunkStore(llm_client=llm, engine=engine)


def test_owner_sees_own_sensitive_and_public(vector_engine) -> None:  # type: ignore[no-untyped-def]
    _seed(vector_engine)
    slugs = {h.slug for h in _store(vector_engine).query("q", top_k=10, caller="alice@bi.group")}
    assert "public-page" in slugs
    assert "private-f-sec" in slugs


def test_other_user_never_sees_sensitive(vector_engine) -> None:  # type: ignore[no-untyped-def]
    _seed(vector_engine)
    slugs = {h.slug for h in _store(vector_engine).query("q", top_k=10, caller="bob@bi.group")}
    assert "public-page" in slugs
    assert "private-f-sec" not in slugs  # ← the leak that must never happen


def test_anonymous_never_sees_sensitive(vector_engine) -> None:  # type: ignore[no-untyped-def]
    _seed(vector_engine)
    slugs = {h.slug for h in _store(vector_engine).query("q", top_k=10)}
    assert "public-page" in slugs
    assert "private-f-sec" not in slugs
