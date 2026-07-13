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


async def test_create_case_defaults_to_sensitive(client: AsyncClient) -> None:
    resp = await client.post("/api/v1/cases", json={"title": "Private by default"})
    assert resp.status_code == 201
    assert resp.json()["sensitive"] is True


async def test_create_case_defaults_to_internal_source(client: AsyncClient) -> None:
    resp = await client.post("/api/v1/cases", json={"title": "Internal by default"})
    assert resp.status_code == 201
    assert resp.json()["source"] == "internal"


async def test_create_case_with_explicit_external_source(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/cases", json={"title": "From a textbook", "source": "external"}
    )
    assert resp.status_code == 201
    assert resp.json()["source"] == "external"


async def test_create_case_rejects_invalid_source(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/v1/cases", json={"title": "Bad source", "source": "sideways"}
    )
    assert resp.status_code == 422


async def test_case_sensitive_flag_persists_across_reload(client: AsyncClient) -> None:
    create_resp = await client.post(
        "/api/v1/cases",
        json={"id": "case-priv-1", "title": "Graduating", "doc_ids": []},
    )
    case_id = create_resp.json()["id"]

    put_resp = await client.put(
        f"/api/v1/cases/{case_id}",
        json={"title": "Graduating", "doc_ids": [], "sensitive": False},
    )
    assert put_resp.status_code == 200

    # Simulate a page reload: fetch the list fresh, as the frontend does on load().
    list_resp = await client.get("/api/v1/cases")
    updated = next(c for c in list_resp.json() if c["id"] == case_id)
    assert updated["sensitive"] is False


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


async def test_similar_cases_empty_when_no_documents(client: AsyncClient) -> None:
    create_resp = await client.post(
        "/api/v1/cases",
        json={"id": "case-sim-1", "title": "No docs yet", "doc_ids": []},
    )
    assert create_resp.status_code == 201

    resp = await client.get("/api/v1/cases/case-sim-1/similar")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_similar_cases_empty_for_unknown_case(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/cases/does-not-exist/similar")
    assert resp.status_code == 200
    assert resp.json() == []
