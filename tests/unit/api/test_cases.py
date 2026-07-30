"""Unit tests for /api/v1/cases CRUD endpoints."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llm_wiki.api.deps import get_db
from llm_wiki.main import app


@pytest_asyncio.fixture
async def client(db_engine) -> AsyncGenerator[AsyncClient, None]:
    session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
        bind=db_engine, expire_on_commit=False, autoflush=False
    )

    async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db

    async with AsyncClient(
        transport=ASGITransport(app=app),  # type: ignore[arg-type]
        base_url="http://test",
    ) as ac:
        yield ac

    app.dependency_overrides.pop(get_db, None)


async def test_list_cases_empty(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/cases")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_create_case(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/cases",
        json={"title": "My Case", "doc_ids": ["doc-1", "doc-2"]},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["title"] == "My Case"
    assert body["doc_ids"] == ["doc-1", "doc-2"]
    assert "id" in body


async def test_list_cases_after_create(client: AsyncClient) -> None:
    await client.post("/api/v1/cases", json={"title": "Alpha", "doc_ids": []})
    await client.post("/api/v1/cases", json={"title": "Beta", "doc_ids": ["d1"]})

    resp = await client.get("/api/v1/cases")
    assert resp.status_code == 200
    cases = resp.json()
    assert len(cases) == 2
    titles = {c["title"] for c in cases}
    assert titles == {"Alpha", "Beta"}


async def test_update_case(client: AsyncClient) -> None:
    create_resp = await client.post(
        "/api/v1/cases",
        json={"id": "case-upd-1", "title": "Original", "doc_ids": ["d1"]},
    )
    assert create_resp.status_code == 201
    case_id = create_resp.json()["id"]

    put_resp = await client.put(
        f"/api/v1/cases/{case_id}",
        json={"title": "Updated Title", "doc_ids": ["d2", "d3"]},
    )
    assert put_resp.status_code == 200
    assert put_resp.json() == {"ok": True}

    list_resp = await client.get("/api/v1/cases")
    updated = next(c for c in list_resp.json() if c["id"] == case_id)
    assert updated["title"] == "Updated Title"
    assert updated["doc_ids"] == ["d2", "d3"]


async def test_delete_case(client: AsyncClient) -> None:
    create_resp = await client.post(
        "/api/v1/cases",
        json={"id": "case-del-1", "title": "To Delete", "doc_ids": []},
    )
    assert create_resp.status_code == 201
    case_id = create_resp.json()["id"]

    del_resp = await client.delete(f"/api/v1/cases/{case_id}")
    assert del_resp.status_code == 200
    assert del_resp.json() == {"ok": True}

    list_resp = await client.get("/api/v1/cases")
    ids = [c["id"] for c in list_resp.json()]
    assert case_id not in ids


async def test_delete_nonexistent_case_returns_404(client: AsyncClient) -> None:
    resp = await client.delete("/api/v1/cases/does-not-exist")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


async def test_update_nonexistent_case_returns_404(client: AsyncClient) -> None:
    resp = await client.put(
        "/api/v1/cases/ghost-id",
        json={"title": "Ghost", "doc_ids": []},
    )
    assert resp.status_code == 404


async def test_list_cases_pagination_and_total_header(client: AsyncClient) -> None:
    for i in range(5):
        await client.post(
            "/api/v1/cases", json={"id": f"case-p{i}", "title": f"Case {i}", "doc_ids": []}
        )

    page1 = await client.get("/api/v1/cases?limit=2&offset=0")
    assert page1.status_code == 200
    assert len(page1.json()) == 2
    assert page1.headers["X-Total-Count"] == "5"

    page2 = await client.get("/api/v1/cases?limit=2&offset=2")
    assert len(page2.json()) == 2
    # pages must not overlap
    ids1 = {c["id"] for c in page1.json()}
    ids2 = {c["id"] for c in page2.json()}
    assert ids1.isdisjoint(ids2)

    # no limit → full list (backward-compatible), header still carries the total
    all_resp = await client.get("/api/v1/cases")
    assert len(all_resp.json()) == 5
    assert all_resp.headers["X-Total-Count"] == "5"


async def test_list_cases_search_by_title(client: AsyncClient) -> None:
    await client.post("/api/v1/cases", json={"id": "c-roof", "title": "Roofing defects"})
    await client.post("/api/v1/cases", json={"id": "c-bud", "title": "Budget review"})

    resp = await client.get("/api/v1/cases?q=roof")  # case-insensitive substring
    titles = [c["title"] for c in resp.json()]
    assert titles == ["Roofing defects"]
    assert resp.headers["X-Total-Count"] == "1"


async def test_list_cases_category_filter(client: AsyncClient) -> None:
    hdr = {"X-User-Email": "alice@bi.group"}
    await client.post(
        "/api/v1/cases", json={"id": "c-pub", "title": "Public one", "sensitive": False}, headers=hdr
    )
    await client.post(
        "/api/v1/cases", json={"id": "c-prv", "title": "Private one", "sensitive": True}, headers=hdr
    )

    all_titles = {c["title"] for c in (await client.get("/api/v1/cases", headers=hdr)).json()}
    assert all_titles == {"Public one", "Private one"}

    pub = (await client.get("/api/v1/cases?category=public", headers=hdr)).json()
    assert {c["title"] for c in pub} == {"Public one"}

    prv = (await client.get("/api/v1/cases?category=private", headers=hdr)).json()
    assert {c["title"] for c in prv} == {"Private one"}


async def test_tags_taxonomy_endpoint(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/tags")
    assert resp.status_code == 200
    names = {t["name"] for t in resp.json()}
    assert len(names) >= 30
    assert {"Качество", "Финансы", "Стратегия"} <= names


async def test_case_tags_persist_and_drop_unknown(client: AsyncClient) -> None:
    # Unknown tags are silently dropped; kept ones come back in taxonomy order.
    resp = await client.post(
        "/api/v1/cases",
        json={"id": "c-tags", "title": "Tagged", "tags": ["Финансы", "BogusTag", "Качество"]},
    )
    assert resp.status_code == 201
    assert resp.json()["tags"] == ["Качество", "Финансы"]

    c = next(x for x in (await client.get("/api/v1/cases")).json() if x["id"] == "c-tags")
    assert c["tags"] == ["Качество", "Финансы"]

    # PUT replaces the tag set (still validated against the taxonomy).
    await client.put(
        "/api/v1/cases/c-tags", json={"title": "Tagged", "doc_ids": [], "tags": ["HR"]}
    )
    c2 = next(x for x in (await client.get("/api/v1/cases")).json() if x["id"] == "c-tags")
    assert c2["tags"] == ["HR"]


async def test_unlink_document_removes_it_from_the_case_and_persists(client: AsyncClient) -> None:
    # item 10: deleting a source must actually unlink it (was a no-op mock).
    await client.post("/api/v1/cases", json={"id": "c-unlink", "title": "C", "doc_ids": ["d1", "d2", "d3"]})
    resp = await client.delete("/api/v1/cases/c-unlink/documents/d2")
    assert resp.status_code == 200

    c = next(x for x in (await client.get("/api/v1/cases")).json() if x["id"] == "c-unlink")
    assert c["doc_ids"] == ["d1", "d3"]  # d2 gone, and it stays gone on reload


async def test_unlink_document_on_missing_case_returns_404(client: AsyncClient) -> None:
    resp = await client.delete("/api/v1/cases/ghost/documents/d1")
    assert resp.status_code == 404


async def test_case_list_exposes_owner(client: AsyncClient) -> None:
    await client.post("/api/v1/cases", json={"id": "c-owner", "title": "C", "doc_ids": []})
    c = next(x for x in (await client.get("/api/v1/cases")).json() if x["id"] == "c-owner")
    assert "owner" in c  # frontend needs it to gate author-only edits


def test_assert_can_edit_only_blocks_a_different_author_under_auth(monkeypatch) -> None:
    # item 4: a public case is author-only — but only enforced with real auth.
    import pytest
    from fastapi import HTTPException

    from llm_wiki.api.v1 import cases
    from llm_wiki.config import settings
    from llm_wiki.storage.metadata import CaseRecord

    owned = CaseRecord(owner="alice@bi.group")
    ownerless = CaseRecord(owner=None)

    # auth OFF (demo): never blocks, even a different caller
    monkeypatch.setattr(settings, "auth_enabled", False)
    cases._assert_can_edit(owned, "bob@bi.group")

    # auth ON: block a different author, allow the author and ownerless cases
    monkeypatch.setattr(settings, "auth_enabled", True)
    with pytest.raises(HTTPException) as exc:
        cases._assert_can_edit(owned, "bob@bi.group")
    assert exc.value.status_code == 403
    cases._assert_can_edit(owned, "alice@bi.group")
    cases._assert_can_edit(ownerless, "bob@bi.group")
