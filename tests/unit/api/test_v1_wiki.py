"""Smoke tests for GET /api/v1/wiki and GET /api/v1/wiki/{slug}/full."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llm_wiki.api.deps import get_db
from llm_wiki.config import settings
from llm_wiki.main import app
from llm_wiki.storage import wiki_store


@pytest_asyncio.fixture
async def client(
    tmp_path: Path,
    db_engine,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncGenerator[AsyncClient, None]:
    """FastAPI test client with overridden DB dependency and temp data dir."""
    object.__setattr__(settings, "data_dir", tmp_path)
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir(exist_ok=True)
    wiki_store.save_page(
        "transformers",
        "Transformers",
        "# Transformers\n\nDeep learning architecture.\n\nSee also [[attention]].\n",
    )
    wiki_store.save_page(
        "attention",
        "Attention",
        "# Attention\n\nUses [[transformers]] mechanisms.\n",
    )

    factory = async_sessionmaker(bind=db_engine, expire_on_commit=False, autoflush=False)

    async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with factory() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db

    async with AsyncClient(
        transport=ASGITransport(app=app),  # type: ignore[arg-type]
        base_url="http://test",
    ) as ac:
        yield ac

    app.dependency_overrides.clear()


async def test_list_wiki_pages(client: AsyncClient) -> None:
    """GET /api/v1/wiki returns summaries for all wiki pages."""
    resp = await client.get("/api/v1/wiki")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    slugs = {item["slug"] for item in data}
    assert slugs == {"transformers", "attention"}
    assert all("snippet" in item for item in data)


async def test_get_wiki_page_full(client: AsyncClient) -> None:
    """GET /api/v1/wiki/{slug}/full returns markdown and backlinks."""
    resp = await client.get("/api/v1/wiki/transformers/full")
    assert resp.status_code == 200
    data = resp.json()
    assert data["slug"] == "transformers"
    assert data["title"] == "Transformers"
    assert "Deep learning architecture" in data["content"]
    assert "attention" in data["backlinks"]


async def test_get_wiki_page_full_not_found(client: AsyncClient) -> None:
    """Unknown slug returns 404."""
    resp = await client.get("/api/v1/wiki/does-not-exist/full")
    assert resp.status_code == 404


async def test_private_page_endpoint_isolation(client: AsyncClient) -> None:
    """A private page is listed/readable only for its owner via the API."""
    wiki_store.save_page(
        "private-hr",
        "HR Secret",
        "# HR Secret\nКонфиденциальный документ.",
        sensitive=True,
        owner="alice@bi.group",
    )

    # Owner: sees it in the list and can open it (flagged sensitive).
    owner_list = await client.get(
        "/api/v1/wiki", headers={"X-User-Email": "alice@bi.group"}
    )
    assert "private-hr" in {p["slug"] for p in owner_list.json()}
    owner_detail = await client.get(
        "/api/v1/wiki/private-hr/full", headers={"X-User-Email": "alice@bi.group"}
    )
    assert owner_detail.status_code == 200
    assert owner_detail.json()["sensitive"] is True

    # Another user: not in the list, and 404 on the detail.
    other_list = await client.get(
        "/api/v1/wiki", headers={"X-User-Email": "bob@bi.group"}
    )
    assert "private-hr" not in {p["slug"] for p in other_list.json()}
    other_detail = await client.get(
        "/api/v1/wiki/private-hr/full", headers={"X-User-Email": "bob@bi.group"}
    )
    assert other_detail.status_code == 404
