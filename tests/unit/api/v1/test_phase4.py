"""Phase 4 integration tests.

Tests:
  1. GET /api/v1/search — library mode JSON response
  2. POST /api/v1/search — expert SSE mode (token order + done event)
  3. POST /api/advisor/ask — rate-limit enforcement
  4. GET /api/v1/documents/{id}/related — related materials via Chroma
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from llm_wiki.api.deps import get_db
from llm_wiki.config import settings
from llm_wiki.llm.embeddings import SearchHit
from llm_wiki.main import app
from llm_wiki.storage.metadata import Base, FileRecord
from llm_wiki.utils.rate_limit import RateLimiter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def db_engine():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(db_engine) -> AsyncGenerator[AsyncSession, None]:
    factory = async_sessionmaker(bind=db_engine, expire_on_commit=False, autoflush=False)
    async with factory() as session:
        yield session


@pytest_asyncio.fixture
async def client(
    tmp_path: Path,
    db_engine,
) -> AsyncGenerator[AsyncClient, None]:
    object.__setattr__(settings, "data_dir", tmp_path)
    (tmp_path / "wiki").mkdir(exist_ok=True)
    (tmp_path / "raw").mkdir(exist_ok=True)

    factory = async_sessionmaker(bind=db_engine, expire_on_commit=False, autoflush=False)

    async def _override() -> AsyncGenerator[AsyncSession, None]:
        async with factory() as sess:
            yield sess

    app.dependency_overrides[get_db] = _override

    async with AsyncClient(
        transport=ASGITransport(app=app),  # type: ignore[arg-type]
        base_url="http://test",
    ) as ac:
        yield ac

    app.dependency_overrides.clear()


async def _insert(
    db_session: AsyncSession,
    file_id: str | None = None,
    original_name: str = "doc.pdf",
    status: str = "DONE",
    created_pages: list[str] | None = None,
) -> FileRecord:
    fid = file_id or f"file-{uuid.uuid4().hex[:8]}"
    record = FileRecord(
        file_id=fid,
        original_name=original_name,
        status=status,
        created_pages=created_pages or [],
        updated_pages=[],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db_session.add(record)
    await db_session.commit()
    return record


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_sse(body: str) -> list[dict]:
    events = []
    for block in body.split("\n\n"):
        block = block.strip()
        if block.startswith("data: "):
            try:
                events.append(json.loads(block[6:]))
            except json.JSONDecodeError:
                pass
    return events


async def _mock_stream(*args, **kwargs) -> AsyncGenerator[str, None]:
    """Async generator: 3 tokens + follow_ups JSON in the last token."""
    for token in ["Hello", " ", 'world {"follow_ups": ["q1?", "q2?"]}']:
        yield token


def _fake_hit(slug: str = "my-slug", score: float = 0.9) -> SearchHit:
    return SearchHit(slug=slug, title="Title", section="Intro", similarity=score)


# ---------------------------------------------------------------------------
# 1. GET /search — library JSON mode
# ---------------------------------------------------------------------------


class TestGetSearch:
    async def test_library_returns_results(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        fr1 = await _insert(db_session, file_id="id1", original_name="alpha.pdf", created_pages=["slug-a"])
        fr2 = await _insert(db_session, file_id="id2", original_name="beta.pdf", created_pages=["slug-b"])

        with patch("llm_wiki.api.v1.search.LLMClient") as MockLLM, \
             patch("llm_wiki.api.v1.search.EmbeddingStore") as MockES:
            llm_inst = MockLLM.return_value
            llm_inst.aclose = AsyncMock()
            es_inst = MockES.return_value
            es_inst.query = MagicMock(return_value=[
                _fake_hit("slug-a", 0.95),
                _fake_hit("slug-b", 0.80),
            ])

            r = await client.get("/api/v1/search?q=test+query")

        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        assert len(data) == 2

        ids = {d["document_id"] for d in data}
        assert "id1" in ids
        assert "id2" in ids

        # Both shape variants (SearchResultRaw + Material)
        for item in data:
            assert "document_title" in item  # SearchResultRaw compat
            assert "title" in item           # Material compat
            assert "document_id" in item

    async def test_empty_query_returns_empty(self, client: AsyncClient) -> None:
        r = await client.get("/api/v1/search?q=")
        assert r.status_code == 200
        assert r.json() == []

    async def test_sql_fallback_when_embedding_fails(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        await _insert(db_session, original_name="my_report.pdf")

        with patch("llm_wiki.api.v1.search.LLMClient") as MockLLM:
            llm_inst = MockLLM.return_value
            llm_inst.aclose = AsyncMock()
            # EmbeddingStore raises on init → semantic search skipped
            with patch("llm_wiki.api.v1.search.EmbeddingStore", side_effect=RuntimeError("no chroma")):
                r = await client.get("/api/v1/search?q=report")

        assert r.status_code == 200
        data = r.json()
        assert len(data) == 1
        assert data[0]["title"] == "my_report"


# ---------------------------------------------------------------------------
# 2. POST /search — expert SSE mode
# ---------------------------------------------------------------------------


class TestPostSearchExpert:
    async def test_expert_tokens_then_done(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        fr1 = await _insert(db_session, file_id="e1", original_name="e1.pdf", created_pages=["slug-e1"])

        with patch("llm_wiki.api.v1.search.LLMClient") as MockLLM, \
             patch("llm_wiki.api.v1.search.EmbeddingStore") as MockES:
            inst = MockLLM.return_value
            inst.aclose = AsyncMock()
            inst.stream_completion = _mock_stream
            MockES.return_value.query = MagicMock(return_value=[_fake_hit("slug-e1", 0.9)])

            r = await client.post(
                "/api/v1/search",
                json={"query": "What is AI?", "mode": "expert", "language": "ru"},
            )

        assert r.status_code == 200
        assert "text/event-stream" in r.headers.get("content-type", "")

        events = _parse_sse(r.text)
        token_events = [e for e in events if "token" in e]
        done_events = [e for e in events if e.get("done")]

        assert len(token_events) >= 1
        assert len(done_events) == 1

        done = done_events[0]
        assert "answer" in done
        assert "materials" in done
        assert "follow_ups" in done
        assert done["follow_ups"] == ["q1?", "q2?"]

        # Tokens arrive before done
        token_idx = [i for i, e in enumerate(events) if "token" in e]
        done_idx = next(i for i, e in enumerate(events) if e.get("done"))
        assert all(ti < done_idx for ti in token_idx)

    async def test_library_mode_single_done_event(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        await _insert(db_session, file_id="l1", original_name="lib.pdf", created_pages=["slug-l1"])

        with patch("llm_wiki.api.v1.search.LLMClient") as MockLLM, \
             patch("llm_wiki.api.v1.search.EmbeddingStore") as MockES:
            inst = MockLLM.return_value
            inst.aclose = AsyncMock()
            MockES.return_value.query = MagicMock(return_value=[_fake_hit("slug-l1", 0.9)])

            r = await client.post(
                "/api/v1/search",
                json={"query": "test", "mode": "library"},
            )

        events = _parse_sse(r.text)
        assert len(events) == 1
        assert events[0]["done"] is True
        assert "results" in events[0]


# ---------------------------------------------------------------------------
# 3. POST /api/advisor/ask — rate limit
# ---------------------------------------------------------------------------


class TestAdvisorRateLimit:
    async def test_rate_limit_after_n_requests(
        self, client: AsyncClient
    ) -> None:
        unique_email = f"rl-{uuid.uuid4().hex}@test.com"
        tight_limiter = RateLimiter(limit=2, window_sec=60)

        async def _tiny_stream(*args, **kwargs) -> AsyncGenerator[str, None]:
            yield "ok"

        with patch("llm_wiki.api.advisor.advisor_limiter", tight_limiter), \
             patch("llm_wiki.api.advisor.LLMClient") as MockLLM, \
             patch("llm_wiki.api.advisor.ChunkStore") as MockCS:
            inst = MockLLM.return_value
            inst.aclose = AsyncMock()
            inst.stream_completion = _tiny_stream
            cs = MockCS.return_value
            from llm_wiki.llm.chunk_store import ChunkHit
            cs.query = MagicMock(return_value=[
                ChunkHit("s", "T", "", 0, "text", 0.9)
            ])

            # First 2 requests within limit
            for _ in range(2):
                r = await client.post(
                    "/api/advisor/ask",
                    json={"query": "What?", "language": "ru"},
                    headers={"X-User-Email": unique_email},
                )
                assert r.status_code == 200
                events = _parse_sse(r.text)
                assert not any("error" in e and "Rate limit" in (e.get("error") or "") for e in events)

            # 3rd request — must be rate-limited
            r = await client.post(
                "/api/advisor/ask",
                json={"query": "What?", "language": "ru"},
                headers={"X-User-Email": unique_email},
            )
            assert r.status_code == 200
            events = _parse_sse(r.text)
            assert len(events) == 1
            assert "error" in events[0]
            assert "Rate limit" in events[0]["error"]


# ---------------------------------------------------------------------------
# 4. GET /documents/{id}/related — Chroma-based related materials
# ---------------------------------------------------------------------------


class TestRelatedMaterials:
    async def test_related_excludes_self_and_returns_others(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        fr1 = await _insert(
            db_session, file_id="main-doc", original_name="main.pdf",
            created_pages=["main-slug"],
        )
        fr2 = await _insert(
            db_session, file_id="rel-doc-1", original_name="rel1.pdf",
            created_pages=["rel-slug-1"],
        )
        fr3 = await _insert(
            db_session, file_id="rel-doc-2", original_name="rel2.pdf",
            created_pages=["rel-slug-2"],
        )

        fake_embeddings = [[0.1, 0.2, 0.3]]
        fake_chroma_results = {
            "ids": [["rel-slug-1#0000", "rel-slug-2#0000"]],
            "metadatas": [[
                {"slug": "rel-slug-1", "title": "Rel1", "section": "", "chunk_idx": 0},
                {"slug": "rel-slug-2", "title": "Rel2", "section": "", "chunk_idx": 0},
            ]],
            "distances": [[0.1, 0.2]],
        }

        with patch("llm_wiki.api.v1.materials.LLMClient") as MockLLM, \
             patch("llm_wiki.api.v1.materials.ChunkStore") as MockCS:
            inst = MockLLM.return_value
            inst.aclose = AsyncMock()
            cs = MockCS.return_value
            cs.get_embeddings_for_slugs = MagicMock(return_value=fake_embeddings)
            cs.count = MagicMock(return_value=3)
            cs._col.query = MagicMock(return_value=fake_chroma_results)

            r = await client.get("/api/v1/documents/main-doc/related")

        assert r.status_code == 200
        data = r.json()
        items = data.get("items", [])
        ids = {m["document_id"] for m in items}

        assert "main-doc" not in ids, "Self must not appear in related"
        assert len(ids) == 2
        assert "rel-doc-1" in ids
        assert "rel-doc-2" in ids

    async def test_related_empty_when_no_slugs(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        await _insert(db_session, file_id="no-wiki", created_pages=[])
        r = await client.get("/api/v1/documents/no-wiki/related")
        assert r.status_code == 200
        assert r.json() == {"items": []}
