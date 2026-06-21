"""API tests for notebook CRUD and scoping (LW-N15 / LW-N16 / LW-N17)."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from llm_wiki.agents.answer import AnswerResult
from llm_wiki.api.deps import get_current_user, get_db
from llm_wiki.api.schemas import CurrentUser
from llm_wiki.main import app
from llm_wiki.storage.metadata import Base, create_notebook, ensure_dev_user


@pytest_asyncio.fixture
async def db_engine():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def client(db_engine) -> AsyncGenerator[AsyncClient, None]:
    factory = async_sessionmaker(bind=db_engine, expire_on_commit=False, autoflush=False)

    async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with factory() as session:
            yield session

    async def _user_a() -> CurrentUser:
        return CurrentUser(id="user-a", name="User A", role="employee")

    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[get_current_user] = _user_a

    async with factory() as session:
        await ensure_dev_user(session)

    async with AsyncClient(
        transport=ASGITransport(app=app),  # type: ignore[arg-type]
        base_url="http://test",
    ) as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client_user_b(db_engine) -> AsyncGenerator[AsyncClient, None]:
    factory = async_sessionmaker(bind=db_engine, expire_on_commit=False, autoflush=False)

    async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with factory() as session:
            yield session

    async def _user_b() -> CurrentUser:
        return CurrentUser(id="user-b", name="User B", role="employee")

    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[get_current_user] = _user_b

    async with AsyncClient(
        transport=ASGITransport(app=app),  # type: ignore[arg-type]
        base_url="http://test",
    ) as ac:
        yield ac

    app.dependency_overrides.clear()


async def test_notebook_crud_owner_scoped(client: AsyncClient) -> None:
    """User can create, list, get, and delete own notebooks."""
    create_resp = await client.post("/api/v1/notebooks", json={"title": "My research"})
    assert create_resp.status_code == 201
    nb = create_resp.json()
    assert nb["title"] == "My research"
    nb_id = nb["id"]

    list_resp = await client.get("/api/v1/notebooks")
    assert list_resp.status_code == 200
    assert len(list_resp.json()) == 1

    get_resp = await client.get(f"/api/v1/notebooks/{nb_id}")
    assert get_resp.status_code == 200

    del_resp = await client.delete(f"/api/v1/notebooks/{nb_id}")
    assert del_resp.status_code == 204

    assert (await client.get(f"/api/v1/notebooks/{nb_id}")).status_code == 404


async def test_notebook_foreign_owner_returns_404(
    client: AsyncClient,
    client_user_b: AsyncClient,
    db_engine,
) -> None:
    """Another user cannot see or delete someone else's notebook."""
    factory = async_sessionmaker(bind=db_engine, expire_on_commit=False, autoflush=False)
    async with factory() as session:
        nb = await create_notebook(session, "user-a", "Secret")

    assert (await client_user_b.get(f"/api/v1/notebooks/{nb.id}")).status_code == 404
    assert (await client_user_b.delete(f"/api/v1/notebooks/{nb.id}")).status_code == 404


async def test_notebook_attach_existing_file(
    client: AsyncClient,
    db_engine,
) -> None:
    """Attach JSON endpoint links an existing file without re-upload."""
    from llm_wiki.storage.metadata import create_file_record

    factory = async_sessionmaker(bind=db_engine, expire_on_commit=False, autoflush=False)
    async with factory() as session:
        nb = await create_notebook(session, "user-a", "Case study")
        await create_file_record(session, "file-abc", "case.pdf")

    with patch("llm_wiki.llm.client.LLMClient") as mock_llm_cls, patch(
        "llm_wiki.llm.chunk_store.ChunkStore"
    ) as mock_chunk_cls:
        mock_chunk = MagicMock()
        mock_chunk.count_by_file_id.return_value = 2
        mock_chunk_cls.return_value = mock_chunk
        mock_llm = MagicMock()
        mock_llm.aclose = AsyncMock()
        mock_llm_cls.return_value = mock_llm

        resp = await client.post(
            f"/api/v1/notebooks/{nb.id}/attach",
            json={"file_id": "file-abc"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["files"]) == 1
    assert body["files"][0]["file_id"] == "file-abc"


async def test_notebook_attach_backfills_chunks_and_ask(
    client: AsyncClient,
    db_engine,
    tmp_path,
) -> None:
    """Attach with zero chunks triggers indexing; ask returns an answer."""
    from llm_wiki.storage.metadata import create_file_record

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "case-1.md").write_text("# Case\n\nSales report quarterly data.", encoding="utf-8")

    factory = async_sessionmaker(bind=db_engine, expire_on_commit=False, autoflush=False)
    async with factory() as session:
        nb = await create_notebook(session, "user-a", "Q&A")
        await create_file_record(session, "case-1", "case-1.md")

    mock_result = AnswerResult(
        answer="Sales data from the report.",
        confidence="high",
        sources=[],
        cost_usd=0.002,
    )
    mock_agent = MagicMock()
    mock_agent.answer_for_notebook = AsyncMock(return_value=mock_result)

    with patch("llm_wiki.orchestrator.pipeline.settings") as mock_settings, patch(
        "llm_wiki.orchestrator.pipeline.index_notebook_source"
    ) as mock_index, patch("llm_wiki.llm.client.LLMClient") as mock_llm_cls, patch(
        "llm_wiki.llm.chunk_store.ChunkStore"
    ) as mock_chunk_cls, patch(
        "llm_wiki.agents.answer.AnswerAgent", return_value=mock_agent
    ):
        mock_settings.raw_dir = raw_dir
        mock_chunk = MagicMock()
        mock_chunk.count_by_file_id.return_value = 0
        mock_chunk_cls.return_value = mock_chunk
        mock_llm = MagicMock()
        mock_llm.aclose = AsyncMock()
        mock_llm_cls.return_value = mock_llm

        attach_resp = await client.post(
            f"/api/v1/notebooks/{nb.id}/attach",
            json={"file_id": "case-1"},
        )

        assert attach_resp.status_code == 200
        mock_index.assert_called_once()

        ask_resp = await client.post(
            f"/api/v1/notebooks/{nb.id}/ask",
            params={"stream": "true"},
            json={"question": "What about sales?", "language": "en"},
        )

    assert ask_resp.status_code == 200
    assert "Sales data" in ask_resp.text
    mock_agent.answer_for_notebook.assert_awaited_once()


async def test_notebook_attach_missing_file_returns_404(
    client: AsyncClient,
    db_engine,
) -> None:
    """Attaching a non-existent file_id returns 404."""
    factory = async_sessionmaker(bind=db_engine, expire_on_commit=False, autoflush=False)
    async with factory() as session:
        nb = await create_notebook(session, "user-a", "Empty")

    with patch("llm_wiki.llm.client.LLMClient") as mock_llm_cls, patch(
        "llm_wiki.llm.chunk_store.ChunkStore"
    ):
        mock_llm = MagicMock()
        mock_llm.aclose = AsyncMock()
        mock_llm_cls.return_value = mock_llm

        resp = await client.post(
            f"/api/v1/notebooks/{nb.id}/attach",
            json={"file_id": "does-not-exist"},
        )

    assert resp.status_code == 404


async def test_notebook_ask_sse_scoped() -> None:
    """Notebook ask streams a done event with scoped answer."""
    mock_result = AnswerResult(
        answer="Scoped answer",
        confidence="high",
        sources=[],
        cost_usd=0.001,
    )
    mock_agent = MagicMock()
    mock_agent.answer_for_notebook = AsyncMock(return_value=mock_result)

    factory = async_sessionmaker(
        bind=create_async_engine("sqlite+aiosqlite:///:memory:", echo=False),
        expire_on_commit=False,
        autoflush=False,
    )
    engine = factory.kw["bind"]
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async def _override_get_db() -> AsyncGenerator[AsyncSession, None]:
        async with factory() as session:
            yield session

    async def _user_a() -> CurrentUser:
        return CurrentUser(id="user-a", name="User A", role="employee")

    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[get_current_user] = _user_a

    async with factory() as session:
        nb = await create_notebook(session, "user-a", "Ask me")

    with patch("llm_wiki.llm.client.LLMClient") as mock_llm_cls, patch(
        "llm_wiki.llm.chunk_store.ChunkStore"
    ), patch("llm_wiki.llm.embeddings.EmbeddingStore"), patch(
        "llm_wiki.agents.answer.AnswerAgent", return_value=mock_agent
    ):
        mock_llm = MagicMock()
        mock_llm.aclose = AsyncMock()
        mock_llm_cls.return_value = mock_llm

        async with AsyncClient(
            transport=ASGITransport(app=app),  # type: ignore[arg-type]
            base_url="http://test",
        ) as ac:
            resp = await ac.post(
                f"/api/v1/notebooks/{nb.id}/ask",
                params={"stream": "true"},
                json={"question": "What is in my sources?", "language": "en"},
            )

    app.dependency_overrides.clear()
    await engine.dispose()

    assert resp.status_code == 200
    assert '"done": true' in resp.text
    assert "Scoped answer" in resp.text
    mock_agent.answer_for_notebook.assert_awaited_once()
