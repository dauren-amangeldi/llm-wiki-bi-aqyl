"""BUG-16: реальный /cases/{id}/similar и батч /twin/sessions (убийца N+1)."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llm_wiki.api.deps import get_db
from llm_wiki.main import app
from llm_wiki.storage.metadata import CaseRecord, ChunkEmbedding, FileRecord, TwinSession

_DIM = 1536


def _vec(direction: int) -> list[float]:
    v = [0.0] * _DIM
    v[direction] = 1.0
    return v


@pytest_asyncio.fixture
async def client(db_engine) -> AsyncGenerator[AsyncClient, None]:
    factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
        bind=db_engine, expire_on_commit=False, autoflush=False
    )

    async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with factory() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db
    async with AsyncClient(
        transport=ASGITransport(app=app),  # type: ignore[arg-type]
        base_url="http://test",
    ) as c:
        yield c
    app.dependency_overrides.clear()


async def test_similar_cases_by_centroid(client: AsyncClient, db_session: AsyncSession) -> None:
    # Кейсы A и B — одинаковое направление векторов (похожи), C — ортогонален.
    for cid, fid, direction in (("case-sa", "f-sa", 0), ("case-sb", "f-sb", 0), ("case-sc", "f-sc", 7)):
        db_session.add(FileRecord(file_id=fid, original_name=f"{fid}.md", status="DONE"))
        db_session.add(ChunkEmbedding(id=f"{fid}#0000", slug=f"pg-{fid}", file_id=fid,
                                      document="txt", embedding=_vec(direction)))
        db_session.add(CaseRecord(id=cid, title=f"Кейс {cid[-2:]}", doc_ids=[fid]))
    await db_session.commit()

    resp = await client.get("/api/v1/cases/case-sa/similar")
    assert resp.status_code == 200
    hits = resp.json()
    assert [h["id"] for h in hits] == ["case-sb"]  # C отфильтрован порогом
    assert hits[0]["similarity_pct"] >= 99

    # Приватный чужой кейс не всплывает.
    db_session.add(FileRecord(file_id="f-priv", original_name="p.md", status="DONE"))
    db_session.add(ChunkEmbedding(id="f-priv#0000", slug="pg-priv", file_id="f-priv",
                                  document="txt", embedding=_vec(0)))
    db_session.add(CaseRecord(id="case-priv", title="Чужой", doc_ids=["f-priv"],
                              sensitive=True, owner="someone@bi.group"))
    await db_session.commit()
    hits = (await client.get("/api/v1/cases/case-sa/similar",
                             headers={"X-User-Email": "alice@bi.group"})).json()
    assert "case-priv" not in [h["id"] for h in hits]


async def test_twin_sessions_batch_visible_only(client: AsyncClient, db_session: AsyncSession) -> None:
    db_session.add(TwinSession(id="tb-1", case_id="case-x", persona_ids=["musk"],
                               created_by="alice@bi.group"))
    db_session.add(TwinSession(id="tb-2", case_id="case-y", persona_ids=["zell"],
                               created_by="bob@bi.group"))
    db_session.add(TwinSession(id="tb-3", case_id="case-x", persona_ids=["dalio"],
                               created_by="demo@bi.group"))  # демо видна всем
    await db_session.commit()

    rows = (await client.get("/api/v1/twin/sessions",
                             headers={"X-User-Email": "alice@bi.group"})).json()
    ids = {r["id"] for r in rows}
    assert ids == {"tb-1", "tb-3"}  # своя + демо; чужая tb-2 скрыта
    assert all("case_id" in r for r in rows)
