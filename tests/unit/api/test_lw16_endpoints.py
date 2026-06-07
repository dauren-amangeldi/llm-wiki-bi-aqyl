"""Unit tests for LW-16 endpoints: GET /wiki/{slug}, GET /log, GET /stats.

Test strategy
-------------
* One shared async fixture creates an **in-memory SQLite** database and
  overrides the ``get_db`` FastAPI dependency, completely isolating the test
  suite from the real ``metadata.db`` on disk.
* ``settings.data_dir`` is patched with ``tmp_path`` so all path properties
  (``wiki_dir``, ``log_path``, ``usage_log_path``, ``issues_path``) resolve
  inside the temporary directory.
* ``httpx.AsyncClient`` with ``ASGITransport`` drives the real ASGI app so
  routing, validation, and content-negotiation all participate.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from llm_wiki.api.deps import get_db
from llm_wiki.config import settings
from llm_wiki.main import app
from llm_wiki.storage.metadata import Base, FileRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def db_engine():
    """Create a fresh in-memory SQLite engine per test."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(db_engine) -> AsyncGenerator[AsyncSession, None]:
    """Yield an async session bound to the in-memory engine."""
    factory = async_sessionmaker(bind=db_engine, expire_on_commit=False, autoflush=False)
    async with factory() as session:
        yield session


@pytest_asyncio.fixture
async def client(
    tmp_path: Path,
    db_engine,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncGenerator[AsyncClient, None]:
    """FastAPI test client with overridden DB dependency and temp data dir."""
    # Redirect all settings paths to tmp_path
    object.__setattr__(settings, "data_dir", tmp_path)
    # Ensure sub-directories exist (mirrors the lifespan setup)
    (tmp_path / "wiki").mkdir(exist_ok=True)
    (tmp_path / "raw").mkdir(exist_ok=True)

    # Override get_db to use the in-memory engine
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


# Helper: insert a FileRecord directly into the test DB
async def _insert_file_record(
    db_session: AsyncSession,
    file_id: str,
    original_name: str = "doc.pdf",
    status: str = "DONE",
    created_pages: list[str] | None = None,
    updated_pages: list[str] | None = None,
    cost_usd: float | None = None,
    created_at: datetime | None = None,
) -> FileRecord:
    record = FileRecord(
        file_id=file_id,
        original_name=original_name,
        status=status,
        created_pages=created_pages or [],
        updated_pages=updated_pages or [],
        cost_usd=cost_usd,
        created_at=created_at or datetime.now(timezone.utc),
    )
    db_session.add(record)
    await db_session.commit()
    return record


# ---------------------------------------------------------------------------
# GET /api/v1/wiki/{slug}
# ---------------------------------------------------------------------------

class TestGetWikiPage:
    async def test_200_json_default(self, client: AsyncClient, tmp_path: Path) -> None:
        content = "# Transformers\n\nBody text.\n\n## Backlinks\n\n- [[gpt]]\n"
        (tmp_path / "wiki" / "transformers.md").write_text(content, encoding="utf-8")

        r = await client.get("/api/v1/wiki/transformers")

        assert r.status_code == 200
        data = r.json()
        assert data["slug"] == "transformers"
        assert data["title"] == "Transformers"
        assert data["backlinks"] == ["gpt"]
        assert data["content"] == content
        assert "last_updated" in data
        # ISO-8601 datetime is parseable
        datetime.fromisoformat(data["last_updated"])

    async def test_200_markdown_accept(self, client: AsyncClient, tmp_path: Path) -> None:
        content = "# Transformers\n\nBody.\n"
        (tmp_path / "wiki" / "transformers.md").write_text(content, encoding="utf-8")

        r = await client.get(
            "/api/v1/wiki/transformers",
            headers={"Accept": "text/markdown"},
        )

        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/markdown")
        assert r.text == content

    async def test_markdown_not_returned_when_json_preferred(
        self, client: AsyncClient, tmp_path: Path
    ) -> None:
        (tmp_path / "wiki" / "page-aa.md").write_text("# Page\n\nBody.", encoding="utf-8")

        r = await client.get(
            "/api/v1/wiki/page-aa",
            headers={"Accept": "application/json"},
        )

        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/json")

    async def test_404_missing_slug(self, client: AsyncClient) -> None:
        r = await client.get("/api/v1/wiki/no-such-page")
        assert r.status_code == 404

    async def test_400_invalid_slug_dotdot(self, client: AsyncClient) -> None:
        r = await client.get("/api/v1/wiki/..%2Fetc")
        # FastAPI decodes the path; the endpoint should reject it
        assert r.status_code in (400, 404)

    async def test_400_invalid_slug_uppercase(self, client: AsyncClient) -> None:
        r = await client.get("/api/v1/wiki/Foo")
        assert r.status_code == 400

    async def test_400_single_char_slug(self, client: AsyncClient) -> None:
        r = await client.get("/api/v1/wiki/x")
        assert r.status_code == 400

    async def test_source_files_populated(
        self, client: AsyncClient, tmp_path: Path, db_session: AsyncSession
    ) -> None:
        content = "# My Page\n\nContent.\n"
        (tmp_path / "wiki" / "my-page.md").write_text(content, encoding="utf-8")

        t1 = datetime.now(timezone.utc) - timedelta(hours=2)
        t2 = datetime.now(timezone.utc) - timedelta(hours=1)
        await _insert_file_record(
            db_session, "file-aaa", created_pages=["my-page"], created_at=t1
        )
        await _insert_file_record(
            db_session, "file-bbb", updated_pages=["my-page"], created_at=t2
        )

        r = await client.get("/api/v1/wiki/my-page")
        assert r.status_code == 200
        source = r.json()["source_files"]
        assert source == ["file-aaa", "file-bbb"]  # chronological order

    async def test_title_fallback_to_slug(
        self, client: AsyncClient, tmp_path: Path
    ) -> None:
        # No # heading — title should be derived from slug
        (tmp_path / "wiki" / "no-title.md").write_text("Just body text.\n", encoding="utf-8")
        r = await client.get("/api/v1/wiki/no-title")
        assert r.status_code == 200
        assert r.json()["title"] == "No Title"


# ---------------------------------------------------------------------------
# GET /api/v1/log
# ---------------------------------------------------------------------------

_LOG_ENTRY_1 = (
    "## 2026-01-01T00:00:00Z — alpha.pdf\n\n"
    "- **File ID**: file-001\n"
    "- **Created**: alpha\n"
    "- **Updated**: none\n"
    "- **Cost**: $0.0050\n"
)
_LOG_ENTRY_2 = (
    "## 2026-02-01T00:00:00Z — beta.pdf\n\n"
    "- **File ID**: file-002\n"
    "- **Created**: beta\n"
    "- **Updated**: none\n"
    "- **Cost**: $0.0060\n"
)
_LOG_ENTRY_3 = (
    "## 2026-03-01T00:00:00Z — gamma.pdf\n\n"
    "- **File ID**: file-003\n"
    "- **Created**: none\n"
    "- **Updated**: gamma\n"
    "- **Cost**: $0.0070\n"
)


class TestGetLog:
    async def test_empty_when_log_absent(self, client: AsyncClient) -> None:
        r = await client.get("/api/v1/log")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 0
        assert data["entries"] == []

    async def test_three_entries_newest_first(
        self, client: AsyncClient, tmp_path: Path
    ) -> None:
        log_content = "# Ingestion Log\n\n" + _LOG_ENTRY_1 + _LOG_ENTRY_2 + _LOG_ENTRY_3
        (tmp_path / "log.md").write_text(log_content, encoding="utf-8")

        r = await client.get("/api/v1/log")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 3
        assert data["page"] == 1
        assert len(data["entries"]) == 3
        # Newest (March) first
        assert "2026-03-01" in data["entries"][0]
        assert "2026-01-01" in data["entries"][2]

    async def test_pagination_page2(
        self, client: AsyncClient, tmp_path: Path
    ) -> None:
        log_content = "# Ingestion Log\n\n" + _LOG_ENTRY_1 + _LOG_ENTRY_2 + _LOG_ENTRY_3
        (tmp_path / "log.md").write_text(log_content, encoding="utf-8")

        r = await client.get("/api/v1/log?page=2&per_page=1")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 3
        assert len(data["entries"]) == 1
        # Page 2 with per_page=1 is the second newest → February
        assert "2026-02-01" in data["entries"][0]

    async def test_per_page_above_max_returns_422(self, client: AsyncClient) -> None:
        r = await client.get("/api/v1/log?per_page=300")
        assert r.status_code == 422

    async def test_page_zero_returns_422(self, client: AsyncClient) -> None:
        r = await client.get("/api/v1/log?page=0")
        assert r.status_code == 422

    async def test_page_beyond_total_returns_empty_entries(
        self, client: AsyncClient, tmp_path: Path
    ) -> None:
        log_content = "# Ingestion Log\n\n" + _LOG_ENTRY_1
        (tmp_path / "log.md").write_text(log_content, encoding="utf-8")

        r = await client.get("/api/v1/log?page=99")
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 1
        assert data["entries"] == []


# ---------------------------------------------------------------------------
# GET /api/v1/stats
# ---------------------------------------------------------------------------

class TestGetStats:
    async def test_empty_data_dir(self, client: AsyncClient) -> None:
        r = await client.get("/api/v1/stats")
        assert r.status_code == 200
        data = r.json()
        assert data["total_files"] == 0
        assert data["total_wiki_pages"] == 0
        assert data["cost_today_usd"] == 0.0
        assert data["cost_this_month_usd"] == 0.0
        assert data["avg_cost_per_ingestion_usd"] == 0.0
        assert data["last_lint_run"] is None

    async def test_with_data(
        self, client: AsyncClient, tmp_path: Path, db_session: AsyncSession
    ) -> None:
        # 3 wiki pages
        for name in ("alpha", "beta", "gamma"):
            (tmp_path / "wiki" / f"{name}.md").write_text(f"# {name}\n")

        # 2 DONE file records
        await _insert_file_record(db_session, "file-1", cost_usd=0.10)
        await _insert_file_record(db_session, "file-2", cost_usd=0.20)

        # Usage log: one entry today, one last month
        now = datetime.now(timezone.utc)
        last_month = now.replace(month=now.month - 1 if now.month > 1 else 12)
        usage_lines = [
            json.dumps({
                "file_id": "file-1",
                "agent_type": "writer",
                "cost_usd": 0.10,
                "timestamp": now.isoformat(),
            }),
            json.dumps({
                "file_id": "file-2",
                "agent_type": "writer",
                "cost_usd": 0.05,
                "timestamp": last_month.isoformat(),
            }),
        ]
        (tmp_path / "usage.log").write_text("\n".join(usage_lines) + "\n")

        # issues.md for last_lint_run
        (tmp_path / "issues.md").write_text("# Issues\n")

        r = await client.get("/api/v1/stats")
        assert r.status_code == 200
        data = r.json()

        assert data["total_files"] == 2
        assert data["total_wiki_pages"] == 3
        assert data["avg_cost_per_ingestion_usd"] == pytest.approx(0.15, abs=1e-4)
        assert data["cost_today_usd"] == pytest.approx(0.10, abs=1e-4)
        assert data["cost_this_month_usd"] == pytest.approx(0.10, abs=1e-4)
        assert data["last_lint_run"] is not None
        # ISO datetime is parseable
        datetime.fromisoformat(data["last_lint_run"])

    async def test_malformed_usage_log_lines_skipped(
        self, client: AsyncClient, tmp_path: Path
    ) -> None:
        (tmp_path / "usage.log").write_text(
            "not json at all\n"
            + json.dumps({"file_id": "x", "cost_usd": 0.01, "timestamp": datetime.now(timezone.utc).isoformat()})
            + "\n"
        )

        r = await client.get("/api/v1/stats")
        assert r.status_code == 200
        # Malformed line skipped; valid line counted
        assert r.json()["cost_today_usd"] == pytest.approx(0.01, abs=1e-4)
