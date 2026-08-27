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


async def test_wiki_title_prefers_stored_title_over_humanised_slug(
    client: AsyncClient,
) -> None:
    """Task 4: an uploaded-file page keyed by an opaque ``private-<uuid>`` slug
    must surface its stored human title, never a hash-like humanised slug — even
    when the body has no H1 to fall back on."""
    wiki_store.save_page(
        "private-019fb341-57d7-727b", "Квартальный отчёт", "Тело без H1-заголовка.\n"
    )
    resp = await client.get("/api/v1/wiki/private-019fb341-57d7-727b/full")
    assert resp.status_code == 200
    title = resp.json()["title"]
    assert title == "Квартальный отчёт"
    assert "019" not in title  # no raw slug/hash leaked into the title


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


async def test_search_strips_frontmatter_and_groups_by_case(
    client, db_session
) -> None:
    """BUG-19: сниппет без YAML-блока; страницы одного кейса схлопываются."""
    from llm_wiki.storage import wiki_store
    from llm_wiki.storage.metadata import CaseRecord, FileRecord

    fm = "---\ntitle: X\ntags: [переговоры, batna]\nsummary: s\n---\n\n"
    wiki_store.save_page("grp-a", "Переговоры: базовый курс", fm + "# A\n\nПереговоры и BATNA — часть один.")
    wiki_store.save_page("grp-b", "Переговоры: продвинутый курс", fm + "# B\n\nПереговоры и ZOPA — часть два.")
    db_session.add(FileRecord(file_id="grp-f1", original_name="a.md", status="DONE", created_pages=["grp-a"]))
    db_session.add(FileRecord(file_id="grp-f2", original_name="b.md", status="DONE", created_pages=["grp-b"]))
    db_session.add(CaseRecord(id="case-grp", title="Тренинг: Переговоры", doc_ids=["grp-f1", "grp-f2"]))
    await db_session.commit()

    hits = (await client.get("/api/v1/wiki?q=переговоры")).json()
    ours = [h for h in hits if h["slug"].startswith("grp-")]
    # Обе страницы кейса схлопнуты в один хит со счётчиком.
    assert len(ours) == 1
    assert ours[0]["case_title"] == "Тренинг: Переговоры"
    assert ours[0]["collapsed_count"] == 1
    # Сниппет чистый: без фронтматтера.
    assert "tags:" not in ours[0]["snippet"] and not ours[0]["snippet"].startswith("---")
