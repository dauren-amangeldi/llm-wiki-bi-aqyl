"""BUG-03 (UAT S1): консультации советника переживают F5.

Сервер ведёт строку консультации по шагам живого флоу: создание генерирует
вопросы, ответы дают «как я понял», бриф присылает клиент. Всё owner-scoped:
чужая консультация — 404 (как у твинов). LLM в тестах замокан.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, patch

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from llm_wiki.api.deps import get_db
from llm_wiki.main import app
from llm_wiki.storage.metadata import AdvisorConsultation

_QUESTIONS = [
    {"id": "goal", "text": "Какова цель?", "options": ["Рост", "Прибыль"], "multi": False},
    {"id": "term", "text": "Горизонт?", "options": ["Год", "Три года"], "multi": False},
]


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


def _patch_questions():
    return patch(
        "llm_wiki.agents.advisor_questions.generate_questions",
        new=AsyncMock(return_value={"decision_type": "market_entry", "questions": _QUESTIONS}),
    )


class _StubLLM:
    """LLMClient для шага ответов: отдаёт готовый пересказ."""

    def __init__(self) -> None: ...

    async def complete(self, **_kw):  # noqa: ANN003
        return json.dumps({"understanding": "Вы решаете, выходить ли на новый рынок."}), None

    async def aclose(self) -> None: ...


async def _start(client: AsyncClient, headers: dict[str, str]) -> str:
    with _patch_questions():
        resp = await client.post(
            "/api/v1/advisor/consultations",
            json={"query": "Стоит ли выходить на рынок Грузии?"},
            headers=headers,
        )
    assert resp.status_code == 201
    data = resp.json()
    assert data["questions"] == _QUESTIONS
    assert data["decision_type_label"]
    return data["id"]


ALICE = {"X-User-Email": "alice@bi.group"}
BOB = {"X-User-Email": "bob@bi.group"}


async def test_full_lifecycle_survives_reload(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    cid = await _start(client, ALICE)

    with patch("llm_wiki.llm.client.LLMClient", _StubLLM):
        resp = await client.post(
            f"/api/v1/advisor/consultations/{cid}/answers",
            json={
                "answers": [{"question": "Какова цель?", "answer": "Рост"}],
                "raw_answers": {"goal": ["Рост"]},
            },
            headers=ALICE,
        )
    assert resp.status_code == 200
    assert "рынок" in resp.json()["understanding"]

    brief = {"headline": "Выходить через партнёра", "situation": "…", "risks": ["валюта"]}
    resp = await client.put(
        f"/api/v1/advisor/consultations/{cid}/brief",
        json={"brief": brief},
        headers=ALICE,
    )
    assert resp.status_code == 200

    # «F5»: полное состояние восстанавливается одним GET.
    resp = await client.get(f"/api/v1/advisor/consultations/{cid}", headers=ALICE)
    assert resp.status_code == 200
    data = resp.json()
    assert data["step"] == "recommendation"
    assert data["brief"] == brief
    assert data["answers"] == {"goal": ["Рост"]}
    assert data["questions"] == _QUESTIONS
    assert "Грузии" in data["situation"]

    resp = await client.post(
        f"/api/v1/advisor/consultations/{cid}/outcome",
        json={"outcome": "decided"},
        headers=ALICE,
    )
    assert resp.status_code == 200

    listed = (await client.get("/api/v1/advisor/consultations", headers=ALICE)).json()
    assert [c["id"] for c in listed] == [cid]
    assert listed[0]["outcome"] == "decided"


async def test_foreign_consultation_is_404(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    cid = await _start(client, ALICE)
    assert (
        await client.get(f"/api/v1/advisor/consultations/{cid}", headers=BOB)
    ).status_code == 404
    assert (
        await client.put(
            f"/api/v1/advisor/consultations/{cid}/brief",
            json={"brief": {}},
            headers=BOB,
        )
    ).status_code == 404
    assert (
        await client.delete(f"/api/v1/advisor/consultations/{cid}", headers=BOB)
    ).status_code == 404
    # Список Боба пуст — чужое не светится.
    assert (await client.get("/api/v1/advisor/consultations", headers=BOB)).json() == []


async def test_invalid_outcome_rejected(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    cid = await _start(client, ALICE)
    resp = await client.post(
        f"/api/v1/advisor/consultations/{cid}/outcome",
        json={"outcome": "whatever"},
        headers=ALICE,
    )
    assert resp.status_code == 422


async def test_delete_own(client: AsyncClient, db_session: AsyncSession) -> None:
    cid = await _start(client, ALICE)
    assert (
        await client.delete(f"/api/v1/advisor/consultations/{cid}", headers=ALICE)
    ).status_code == 200
    db_session.expire_all()
    assert await db_session.get(AdvisorConsultation, cid) is None
