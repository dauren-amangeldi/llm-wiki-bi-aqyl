"""BUG-01 (UAT S1): выбор источников в чате кейса должен уважаться.

Репро ревизии: «Снять все» → «Контекст: 0 из N» → вопрос → полный ответ по
снятому документу. Причина — [] терялся дважды (falsy на фронте и на бэке).
Контракт после фикса:
  doc_ids отсутствует → ответ по всем документам кейса (совместимость);
  doc_ids: []        → честный отказ БЕЗ вызова LLM;
  doc_ids: [чужие]   → тоже отказ, а не молчаливое расширение до всего кейса;
  doc_ids: [часть]   → ретривер видит только выбранные документы.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from types import SimpleNamespace
from unittest.mock import patch

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llm_wiki.api.deps import get_db
from llm_wiki.main import app
from llm_wiki.storage.metadata import CaseRecord, FileRecord


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


async def _seed_case(db: AsyncSession, n_files: int = 2) -> tuple[str, list[str]]:
    file_ids = []
    for i in range(n_files):
        fid = f"f-src-{i}"
        db.add(
            FileRecord(
                file_id=fid, original_name=f"doc{i}.md", status="DONE",
                created_pages=[f"page-{i}"],
            )
        )
        file_ids.append(fid)
    db.add(CaseRecord(id="case-sel", title="Выбор источников", doc_ids=file_ids))
    await db.commit()
    return "case-sel", file_ids


class _NoLLM:
    """LLMClient, само создание которого валит тест — отказ не должен стоить денег."""

    def __init__(self) -> None:  # pragma: no cover — сам факт вызова = провал
        raise AssertionError("LLM must not be constructed for a refusal answer")


def _fake_agent(captured: dict):
    """AnswerAgent-стаб: запоминает, какие документы дошли до ретривера."""

    class _Agent:
        def __init__(self, *a, **kw) -> None: ...

        async def answer_for_case(self, *, question, documents, language, file_id, title=""):
            captured["documents"] = [d.file_id for d in documents]
            return SimpleNamespace(
                answer="ok",
                confidence="high",
                sources=[SimpleNamespace(slug="page-0", title="Doc 0", quote=None)],
                cost_usd=0.0,
            )

    return _Agent


async def test_empty_selection_refuses_without_llm(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    case_id, _ = await _seed_case(db_session)
    with patch("llm_wiki.llm.client.LLMClient", _NoLLM):
        resp = await client.post(
            f"/api/v1/cases/{case_id}/ask",
            json={"question": "О чём материалы?", "doc_ids": []},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["insufficient_evidence"] is True
    assert data["citations"] == []
    assert "сточник" in data["answer"]  # «Источники не выбраны…»


async def test_foreign_ids_refuse_instead_of_widening(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    case_id, _ = await _seed_case(db_session)
    with patch("llm_wiki.llm.client.LLMClient", _NoLLM):
        resp = await client.post(
            f"/api/v1/cases/{case_id}/ask",
            json={"question": "Вопрос", "doc_ids": ["file-from-another-case"]},
        )
    assert resp.status_code == 200
    assert resp.json()["insufficient_evidence"] is True
    assert resp.json()["citations"] == []


async def test_partial_selection_narrows_retrieval(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    case_id, file_ids = await _seed_case(db_session, n_files=3)
    captured: dict = {}
    with patch("llm_wiki.agents.answer.AnswerAgent", _fake_agent(captured)):
        resp = await client.post(
            f"/api/v1/cases/{case_id}/ask",
            json={"question": "Вопрос", "doc_ids": [file_ids[1]]},
        )
    assert resp.status_code == 200
    assert captured["documents"] == [file_ids[1]]
    assert resp.json()["citations"][0]["case_id"] == case_id


async def test_absent_field_means_all_documents(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    case_id, file_ids = await _seed_case(db_session, n_files=2)
    captured: dict = {}
    with patch("llm_wiki.agents.answer.AnswerAgent", _fake_agent(captured)):
        resp = await client.post(
            f"/api/v1/cases/{case_id}/ask",
            json={"question": "Вопрос"},  # поле doc_ids не прислано вовсе
        )
    assert resp.status_code == 200
    assert sorted(captured["documents"]) == sorted(file_ids)
    # Цитата несёт физический файл-источник страницы (page-0 создана f-src-0).
    assert resp.json()["citations"][0]["file_name"] == "doc0"
