"""Phase 3 integration tests — POST /documents/{id}/ask chat endpoint.

Covers:
  - Non-streaming (JSON) path — happy path
  - Non-streaming — insufficient evidence (no chunks)
  - Non-streaming — insufficient evidence (no wiki slugs)
  - Streaming (SSE) path — token order + done event
  - Streaming — insufficient evidence path
  - /cards/{id}/ask alias

All LLM and Chroma I/O is mocked; tests run with in-memory SQLite.
"""

from __future__ import annotations

import json
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
from llm_wiki.llm.chunk_store import ChunkHit
from llm_wiki.main import app
from llm_wiki.storage.metadata import Base, FileRecord


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
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncGenerator[AsyncClient, None]:
    object.__setattr__(settings, "data_dir", tmp_path)
    (tmp_path / "wiki").mkdir(exist_ok=True)
    (tmp_path / "raw").mkdir(exist_ok=True)

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


async def _insert(
    db_session: AsyncSession,
    file_id: str = "file-001",
    original_name: str = "my-doc.pdf",
    status: str = "DONE",
    created_pages: list[str] | None = None,
) -> FileRecord:
    record = FileRecord(
        file_id=file_id,
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


def _make_fake_chunk(idx: int = 0) -> ChunkHit:
    return ChunkHit(
        slug="my-doc",
        title="My Doc",
        section="Intro",
        chunk_idx=idx,
        text=f"Relevant content about the topic {idx}.",
        similarity=0.9 - idx * 0.05,
    )


def _fake_complete_response(text: str = "Great answer [chunk_0].\n") -> str:
    follow_ups_json = '{"follow_ups": ["Follow up 1?", "Follow up 2?", "Follow up 3?"]}'
    return text + "\n\n" + follow_ups_json


async def _mock_stream(*args: object, **kwargs: object) -> AsyncGenerator[str, None]:
    """Async generator that simulates streaming tokens."""
    tokens = [
        "The answer is ",
        "42 [chunk_0]. ",
        '{"follow_ups": ["Q1?", "Q2?"]}',
    ]
    for token in tokens:
        yield token


def _parse_sse(body: str) -> list[dict]:
    """Parse SSE body into a list of event dicts."""
    events = []
    for block in body.split("\n\n"):
        block = block.strip()
        if block.startswith("data: "):
            try:
                events.append(json.loads(block[6:]))
            except json.JSONDecodeError:
                pass
    return events


# ---------------------------------------------------------------------------
# Non-streaming (JSON) path
# ---------------------------------------------------------------------------


class TestAskDocumentJSON:
    async def test_happy_path_returns_answer(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        await _insert(db_session, created_pages=["my-doc"])

        with (
            patch("llm_wiki.api.v1.chat.LLMClient") as MockLLM,
            patch("llm_wiki.api.v1.chat.ChunkStore") as MockCS,
        ):
            inst = MockLLM.return_value
            inst.aclose = AsyncMock()
            inst.complete = AsyncMock(
                return_value=(_fake_complete_response("Great answer [chunk_0]."), MagicMock())
            )
            cs = MockCS.return_value
            cs.query = MagicMock(return_value=[_make_fake_chunk()])

            r = await client.post(
                "/api/v1/documents/file-001/ask",
                json={"question": "What is this?"},
            )

        assert r.status_code == 200
        data = r.json()
        assert "answer" in data
        assert data["answer"] != ""
        assert "citations" in data
        assert "follow_ups" in data
        assert len(data["follow_ups"]) > 0
        assert data.get("insufficient_evidence") is not True

    async def test_citations_extracted(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        await _insert(db_session, created_pages=["my-doc"])

        with (
            patch("llm_wiki.api.v1.chat.LLMClient") as MockLLM,
            patch("llm_wiki.api.v1.chat.ChunkStore") as MockCS,
        ):
            inst = MockLLM.return_value
            inst.aclose = AsyncMock()
            inst.complete = AsyncMock(
                return_value=(
                    _fake_complete_response("See [chunk_0] and also [chunk_1]."),
                    MagicMock(),
                )
            )
            cs = MockCS.return_value
            cs.query = MagicMock(return_value=[_make_fake_chunk(0), _make_fake_chunk(1)])

            r = await client.post(
                "/api/v1/documents/file-001/ask",
                json={"question": "Citations?"},
            )

        citations = r.json()["citations"]
        anchors = {c["anchor"] for c in citations}
        assert "chunk_0" in anchors
        assert "chunk_1" in anchors

    async def test_insufficient_evidence_no_slugs(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        # Document exists but ingestion produced no wiki pages
        await _insert(db_session, created_pages=[])

        r = await client.post(
            "/api/v1/documents/file-001/ask",
            json={"question": "Anything?"},
        )

        assert r.status_code == 200
        data = r.json()
        assert data["insufficient_evidence"] is True
        assert data["contact"] == "knowledge-team@bi.group"

    async def test_insufficient_evidence_no_chunks(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        await _insert(db_session, created_pages=["my-doc"])

        with (
            patch("llm_wiki.api.v1.chat.LLMClient") as MockLLM,
            patch("llm_wiki.api.v1.chat.ChunkStore") as MockCS,
        ):
            inst = MockLLM.return_value
            inst.aclose = AsyncMock()
            cs = MockCS.return_value
            cs.query = MagicMock(return_value=[])  # Chroma has no chunks for this doc

            r = await client.post(
                "/api/v1/documents/file-001/ask",
                json={"question": "Irrelevant question"},
            )

        assert r.status_code == 200
        data = r.json()
        assert data["insufficient_evidence"] is True

    async def test_404_missing_document(self, client: AsyncClient) -> None:
        r = await client.post(
            "/api/v1/documents/nonexistent/ask",
            json={"question": "Hello?"},
        )
        # Missing document → treated as insufficient_evidence (not 404 in JSON mode)
        assert r.status_code == 200
        assert r.json()["insufficient_evidence"] is True

    async def test_language_and_mode_passed(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        await _insert(db_session, created_pages=["my-doc"])

        with (
            patch("llm_wiki.api.v1.chat.LLMClient") as MockLLM,
            patch("llm_wiki.api.v1.chat.ChunkStore") as MockCS,
        ):
            inst = MockLLM.return_value
            inst.aclose = AsyncMock()
            inst.complete = AsyncMock(
                return_value=(_fake_complete_response("OK."), MagicMock())
            )
            cs = MockCS.return_value
            cs.query = MagicMock(return_value=[_make_fake_chunk()])

            r = await client.post(
                "/api/v1/documents/file-001/ask",
                json={"question": "Q?", "language": "en", "mode": "library"},
            )

        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Streaming (SSE) path
# ---------------------------------------------------------------------------


class TestAskDocumentSSE:
    async def test_sse_token_then_done(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        await _insert(db_session, created_pages=["my-doc"])

        with (
            patch("llm_wiki.api.v1.chat.LLMClient") as MockLLM,
            patch("llm_wiki.api.v1.chat.ChunkStore") as MockCS,
        ):
            inst = MockLLM.return_value
            inst.aclose = AsyncMock()
            inst.stream_completion = _mock_stream
            cs = MockCS.return_value
            cs.query = MagicMock(return_value=[_make_fake_chunk()])

            r = await client.post(
                "/api/v1/documents/file-001/ask?stream=true",
                json={"question": "Stream this!"},
            )

        assert r.status_code == 200
        assert "text/event-stream" in r.headers.get("content-type", "")

        events = _parse_sse(r.text)
        token_events = [e for e in events if "token" in e]
        done_events = [e for e in events if e.get("done")]

        assert len(token_events) > 0, "Expected at least one token event"
        assert len(done_events) == 1, "Expected exactly one done event"

        # Tokens arrive before the done event
        token_indices = [i for i, e in enumerate(events) if "token" in e]
        done_index = next(i for i, e in enumerate(events) if e.get("done"))
        assert all(ti < done_index for ti in token_indices)

    async def test_sse_done_event_fields(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        await _insert(db_session, created_pages=["my-doc"])

        with (
            patch("llm_wiki.api.v1.chat.LLMClient") as MockLLM,
            patch("llm_wiki.api.v1.chat.ChunkStore") as MockCS,
        ):
            inst = MockLLM.return_value
            inst.aclose = AsyncMock()
            inst.stream_completion = _mock_stream
            cs = MockCS.return_value
            cs.query = MagicMock(return_value=[_make_fake_chunk()])

            r = await client.post(
                "/api/v1/documents/file-001/ask?stream=true",
                json={"question": "Fields?"},
            )

        events = _parse_sse(r.text)
        done = next(e for e in events if e.get("done"))
        assert "answer" in done
        assert "sources" in done
        assert "citations" in done
        assert "follow_ups" in done
        # follow_ups parsed from _mock_stream JSON block
        assert len(done["follow_ups"]) > 0
        # citations parsed from [chunk_0] in answer
        assert any(c["anchor"] == "chunk_0" for c in done["citations"])

    async def test_sse_insufficient_no_slugs(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        await _insert(db_session, created_pages=[])

        r = await client.post(
            "/api/v1/documents/file-001/ask?stream=true",
            json={"question": "Anything?"},
        )

        assert r.status_code == 200
        events = _parse_sse(r.text)
        done_events = [e for e in events if e.get("done")]
        assert len(done_events) == 1
        assert done_events[0].get("insufficient_evidence") is True

    async def test_sse_insufficient_no_chunks(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        await _insert(db_session, created_pages=["my-doc"])

        with (
            patch("llm_wiki.api.v1.chat.LLMClient") as MockLLM,
            patch("llm_wiki.api.v1.chat.ChunkStore") as MockCS,
        ):
            inst = MockLLM.return_value
            inst.aclose = AsyncMock()
            cs = MockCS.return_value
            cs.query = MagicMock(return_value=[])

            r = await client.post(
                "/api/v1/documents/file-001/ask?stream=true",
                json={"question": "Off-topic?"},
            )

        assert r.status_code == 200
        events = _parse_sse(r.text)
        done = next((e for e in events if e.get("done")), None)
        assert done is not None
        assert done.get("insufficient_evidence") is True


# ---------------------------------------------------------------------------
# /cards/{card_id}/ask  alias
# ---------------------------------------------------------------------------


class TestAskCard:
    async def test_card_alias_works(
        self, client: AsyncClient, db_session: AsyncSession
    ) -> None:
        await _insert(db_session, file_id="card-001", created_pages=["my-doc"])

        with (
            patch("llm_wiki.api.v1.chat.LLMClient") as MockLLM,
            patch("llm_wiki.api.v1.chat.ChunkStore") as MockCS,
        ):
            inst = MockLLM.return_value
            inst.aclose = AsyncMock()
            inst.complete = AsyncMock(
                return_value=(_fake_complete_response("Card answer."), MagicMock())
            )
            cs = MockCS.return_value
            cs.query = MagicMock(return_value=[_make_fake_chunk()])

            r = await client.post(
                "/api/v1/cards/card-001/ask",
                json={"question": "Card question?"},
            )

        assert r.status_code == 200
        assert "answer" in r.json()
