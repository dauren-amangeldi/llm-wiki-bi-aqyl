"""Unit tests for the API layer — POST /files endpoint (LW-5 + LW-19)."""

import io
import json
from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llm_wiki.api.rate_limit import InMemoryRateLimiter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def test_app(tmp_path: Path, db_engine):  # type: ignore[misc]
    """FastAPI app wired to a temp DB and temp data directories."""
    from llm_wiki.main import app
    import llm_wiki.api.deps as deps_module

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    # Override the DB session dependency
    session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
        bind=db_engine, expire_on_commit=False, autoflush=False
    )

    async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[deps_module.get_db] = _override_get_db

    yield app, tmp_path, raw_dir

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_upload(content: bytes, filename: str) -> dict:  # type: ignore[type-arg]
    """Build the files dict for httpx multipart upload."""
    return {"file": (filename, io.BytesIO(content), "application/octet-stream")}


def _mock_settings(raw_dir: Path) -> MagicMock:
    """Return a mock settings with raw_dir pointing at tmp_path."""
    m = MagicMock()
    m.raw_dir = raw_dir
    m.max_file_size_mb = 50
    m.allowed_extensions = frozenset({".pdf", ".md"})
    m.service_name = "llm-wiki-test"
    return m


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_health_check(test_app: tuple) -> None:  # type: ignore[type-arg]
    """GET /health returns 200."""
    app, *_ = test_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_healthz_liveness(test_app: tuple) -> None:  # type: ignore[type-arg]
    """GET /healthz (liveness alias) returns 200 without touching dependencies."""
    app, *_ = test_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_readyz_reports_all_dependencies(test_app: tuple) -> None:  # type: ignore[type-arg]
    """GET /readyz returns a per-dependency breakdown and the right status code.

    The local object store (temp dir) is always reachable; external services may
    be down in the unit environment, so we assert on structure + status coupling
    rather than a fixed 200.
    """
    app, *_ = test_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/readyz")

    body = resp.json()
    assert set(body["checks"]) == {"postgres", "redis", "object_store"}
    assert body["checks"]["object_store"] == "ok"
    ready = all(v == "ok" for v in body["checks"].values())
    assert body["status"] == ("ready" if ready else "not_ready")
    assert resp.status_code == (200 if ready else 503)


async def test_upload_pdf_returns_202(test_app: tuple, tmp_path: Path) -> None:  # type: ignore[type-arg]
    """POST /files with a valid PDF returns 202 with file_id and task_id."""
    app, data_tmp, raw_dir = test_app
    fake_pdf = b"%PDF-1.4 fake content that is long enough"

    mock_settings = _mock_settings(raw_dir)

    with (
        patch("llm_wiki.api.routes.process_file_task") as mock_task,
        patch("llm_wiki.api.routes.settings", mock_settings),
    ):
        mock_task.delay.return_value = MagicMock(id="celery-task-123")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/files", files=_make_upload(fake_pdf, "report.pdf"))

    assert resp.status_code == 202
    body = resp.json()
    assert "file_id" in body
    assert body["task_id"] == "celery-task-123"
    assert body["status"] == "queued"


async def test_upload_md_returns_202(test_app: tuple, tmp_path: Path) -> None:  # type: ignore[type-arg]
    """POST /files with a valid Markdown file returns 202."""
    app, data_tmp, raw_dir = test_app
    md_content = b"# Hello\n\nSome content here.\n"
    mock_settings = _mock_settings(raw_dir)

    with (
        patch("llm_wiki.api.routes.process_file_task") as mock_task,
        patch("llm_wiki.api.routes.settings", mock_settings),
    ):
        mock_task.delay.return_value = MagicMock(id="celery-md-456")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/files", files=_make_upload(md_content, "notes.md"))

    assert resp.status_code == 202
    assert resp.json()["status"] == "queued"


async def test_upload_unsupported_type_returns_400(test_app: tuple) -> None:  # type: ignore[type-arg]
    """POST /files with a .txt file returns 400."""
    app, _, raw_dir = test_app
    mock_settings = _mock_settings(raw_dir)

    with (
        patch("llm_wiki.api.routes.process_file_task"),
        patch("llm_wiki.api.routes.settings", mock_settings),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/files", files=_make_upload(b"plain text", "readme.txt")
            )
    assert resp.status_code == 400
    assert "Unsupported file type" in resp.json()["detail"]


async def test_upload_too_large_returns_413(test_app: tuple) -> None:  # type: ignore[type-arg]
    """POST /files with a file over 50 MB returns 413."""
    app, _, raw_dir = test_app
    mock_settings = _mock_settings(raw_dir)
    # Content just over the 50 MB limit
    oversized = b"x" * (51 * 1024 * 1024)

    with (
        patch("llm_wiki.api.routes.process_file_task"),
        patch("llm_wiki.api.routes.settings", mock_settings),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/files", files=_make_upload(oversized, "big.pdf")
            )
    assert resp.status_code == 413


async def test_upload_saves_file_to_raw(test_app: tuple) -> None:  # type: ignore[type-arg]
    """Uploaded file must be saved to raw_dir with file_id as stem."""
    app, data_tmp, raw_dir = test_app
    content = b"%PDF-1.4 test"
    mock_settings = _mock_settings(raw_dir)

    with (
        patch("llm_wiki.api.routes.process_file_task") as mock_task,
        patch("llm_wiki.api.routes.settings", mock_settings),
    ):
        mock_task.delay.return_value = MagicMock(id="t1")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/files", files=_make_upload(content, "doc.pdf"))

    from datetime import datetime, timezone

    from llm_wiki.storage.object_store import get_object_store, raw_key

    file_id = resp.json()["file_id"]
    store = get_object_store()
    # Upload stores under a date-partitioned key (YYYY/MM/DD/<file_id>.pdf);
    # recompute it with today's UTC date to locate the object.
    key = raw_key(file_id, ".pdf", datetime.now(timezone.utc))
    assert store.exists(key)
    assert store.get_bytes(key) == content


async def test_upload_creates_db_record(test_app: tuple, db_engine) -> None:  # type: ignore[type-arg, misc]
    """Uploading a file creates a FileRecord in the database."""
    app, data_tmp, raw_dir = test_app
    content = b"# Markdown content"
    mock_settings = _mock_settings(raw_dir)

    with (
        patch("llm_wiki.api.routes.process_file_task") as mock_task,
        patch("llm_wiki.api.routes.settings", mock_settings),
    ):
        mock_task.delay.return_value = MagicMock(id="t2")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/files", files=_make_upload(content, "page.md"))

    file_id = resp.json()["file_id"]

    from llm_wiki.storage.metadata import get_file_record

    session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
        bind=db_engine, expire_on_commit=False
    )
    async with session_factory() as session:
        record = await get_file_record(session, file_id)

    assert record is not None
    assert record.original_name == "page.md"
    assert record.status == "RECEIVED"


# ---------------------------------------------------------------------------
# GET /files/{file_id}
# ---------------------------------------------------------------------------


async def test_get_file_status_not_found(test_app: tuple) -> None:  # type: ignore[type-arg]
    """GET /files/{file_id} returns 404 for an unknown file_id."""
    app, _, _raw_dir = test_app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/files/nonexistent-id")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


async def test_get_file_status_returns_record(test_app: tuple) -> None:  # type: ignore[type-arg]
    """GET /files/{file_id} returns full status after a successful upload."""
    app, _, raw_dir = test_app
    mock_settings = _mock_settings(raw_dir)

    # Upload a file to create the DB record
    with (
        patch("llm_wiki.api.routes.process_file_task") as mock_task,
        patch("llm_wiki.api.routes.settings", mock_settings),
    ):
        mock_task.delay.return_value = MagicMock(id="task-1")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            upload_resp = await client.post(
                "/api/v1/files",
                files=_make_upload(b"%PDF-1.4 test", "report.pdf"),
            )

    assert upload_resp.status_code == 202
    file_id = upload_resp.json()["file_id"]

    # Query status — no pipeline ran, so history/pages/cost are empty/null
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(f"/api/v1/files/{file_id}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["file_id"] == file_id
    assert body["original_name"] == "report.pdf"
    assert body["status"] == "RECEIVED"
    assert body["state_history"] == []
    assert body["created_pages"] == []
    assert body["updated_pages"] == []
    assert body["cost_usd"] is None


# ---------------------------------------------------------------------------
# LW-19: rate limiting, kill switch, budget check
# ---------------------------------------------------------------------------


def _mock_settings_lw19(raw_dir: Path, **overrides: object) -> MagicMock:
    """Mock settings with LW-19 fields explicitly set."""
    m = MagicMock()
    m.raw_dir = raw_dir
    m.max_file_size_mb = 50
    m.allowed_extensions = frozenset({".pdf", ".md"})
    m.service_name = "llm-wiki-test"
    m.ingestion_enabled = True
    m.ingestion_rate_limit_per_min = 10
    m.daily_cost_limit_usd = None
    m.daily_token_limit = None
    for k, v in overrides.items():
        setattr(m, k, v)
    return m


async def test_rate_limit_returns_429_after_limit(test_app: tuple) -> None:  # type: ignore[type-arg]
    """POST /files returns 429 with Retry-After when per-IP limit is exceeded."""
    from llm_wiki.api import deps as deps_module

    app, data_tmp, raw_dir = test_app
    mock_settings = _mock_settings_lw19(raw_dir)

    # Install a tight rate limiter (max 2 requests)
    limiter = InMemoryRateLimiter(max_requests=2, window_seconds=60)

    def _get_tight_limiter() -> InMemoryRateLimiter:
        return limiter

    app.dependency_overrides[deps_module.get_files_rate_limiter] = _get_tight_limiter
    try:
        with (
            patch("llm_wiki.api.routes.process_file_task") as mock_task,
            patch("llm_wiki.api.routes.settings", mock_settings),
        ):
            mock_task.delay.return_value = MagicMock(id="t-rl")
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                # First two should succeed
                r1 = await client.post("/api/v1/files", files=_make_upload(b"%PDF-1.4 ok", "a.pdf"))
                r2 = await client.post("/api/v1/files", files=_make_upload(b"%PDF-1.4 ok", "b.pdf"))
                # Third should be rejected
                r3 = await client.post("/api/v1/files", files=_make_upload(b"%PDF-1.4 ok", "c.pdf"))

        assert r1.status_code == 202
        assert r2.status_code in (202, 200)  # second may be a duplicate
        assert r3.status_code == 429
        assert "Retry-After" in r3.headers
    finally:
        app.dependency_overrides.pop(deps_module.get_files_rate_limiter, None)


async def test_kill_switch_returns_503(test_app: tuple) -> None:  # type: ignore[type-arg]
    """POST /files returns 503 when INGESTION_ENABLED=false."""
    app, data_tmp, raw_dir = test_app
    mock_settings = _mock_settings_lw19(raw_dir, ingestion_enabled=False)

    with (
        patch("llm_wiki.api.routes.process_file_task"),
        patch("llm_wiki.api.routes.settings", mock_settings),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/files", files=_make_upload(b"%PDF-1.4", "x.pdf"))

    assert resp.status_code == 503
    assert "disabled" in resp.json()["detail"].lower()


async def test_budget_exceeded_returns_503(test_app: tuple, tmp_path: Path) -> None:  # type: ignore[type-arg]
    """POST /files returns 503 when daily cost limit is already exceeded."""
    app, data_tmp, raw_dir = test_app

    # Create a usage.log that shows $2.00 already spent today (dynamic date so
    # the "today" budget window always matches the test run).
    from datetime import datetime, timezone

    usage_log = tmp_path / "usage.log"
    record = {
        "file_id": "prev",
        "agent_type": "writer",
        "model": "gpt-5.4-mini",
        "input_tokens": 1000,
        "output_tokens": 500,
        "cached_input_tokens": 0,
        "cost_usd": 2.00,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_ms": 300,
    }
    usage_log.write_text(json.dumps(record) + "\n")

    mock_settings = _mock_settings_lw19(
        raw_dir,
        daily_cost_limit_usd=1.0,   # limit is $1.00, already spent $2.00
        daily_token_limit=None,
    )
    mock_settings.usage_log_path = usage_log

    with (
        patch("llm_wiki.api.routes.process_file_task"),
        patch("llm_wiki.api.routes.settings", mock_settings),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/api/v1/files", files=_make_upload(b"%PDF-1.4", "y.pdf"))

    assert resp.status_code == 503
    assert "budget" in resp.json()["detail"].lower()
