"""Unit tests for AdvisorAgent (LW-N7 / LW-N9)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_wiki.agents.advisor import AdvisorAgent, AdvisorResponse
from llm_wiki.llm.chunk_store import ChunkHit, ChunkStore
from llm_wiki.llm.client import LLMClient


def _mock_llm(json_payload: dict) -> LLMClient:  # type: ignore[type-arg]
    mock = MagicMock(spec=LLMClient)
    mock.load_prompt.return_value = "prompt"
    usage = MagicMock()
    usage.cost_usd = 0.0031
    mock.complete = AsyncMock(return_value=(json.dumps(json_payload), usage))
    return mock  # type: ignore[return-value]


def _mock_chunk_store(hits: list[ChunkHit]) -> ChunkStore:
    mock = MagicMock(spec=ChunkStore)
    mock.query.return_value = hits
    return mock  # type: ignore[return-value]


def _hit(
    *,
    file_id: str = "case-001",
    slug: str = "lean-project",
    similarity: float = 0.82,
    text: str = "Сроки этапа сократились на 12%.",
) -> ChunkHit:
    return ChunkHit(
        slug=slug,
        title="Lean Project",
        section="Results",
        chunk_idx=0,
        text=text,
        similarity=similarity,
        file_id=file_id,
    )


@pytest.mark.asyncio
async def test_advisor_queries_globally_not_by_usage_file_id() -> None:
    """Advisor must not filter Chroma by the usage correlation id."""
    hits = [_hit()]
    chunk_store = _mock_chunk_store(hits)
    llm = _mock_llm(
        {
            "title": "T",
            "summary": "S",
            "points": [
                {
                    "heading": "H",
                    "body": "B",
                    "metric": "12%",
                    "tag": "Tag",
                    "case_id": "case-001",
                }
            ],
            "source": "src",
            "caseCount": 1,
        }
    )
    agent = AdvisorAgent(llm, chunk_store)

    await agent.advise("question?", file_id="advisor")

    chunk_store.query.assert_called_once_with(
        "question?", top_k=8, usage_file_id="advisor"
    )


@pytest.mark.asyncio
async def test_advisor_happy_path_returns_structured_response() -> None:
    hits = [
        _hit(file_id="case-001"),
        _hit(file_id="case-002", slug="bim-model", text="BIM снизил коллизии на 60%."),
    ]
    llm = _mock_llm(
        {
            "title": "Практические шаги",
            "summary": "Два кейса показывают сокращение сроков и коллизий.",
            "points": [
                {
                    "heading": "Lean-планирование",
                    "body": "На объекте применили lean-подход.",
                    "metric": "12%",
                    "tag": "Сроки",
                    "case_id": "case-001",
                },
                {
                    "heading": "BIM-модель",
                    "body": "BIM помог снизить коллизии.",
                    "metric": "60%",
                    "tag": "Стоимость",
                    "case_id": "case-002",
                },
            ],
            "source": "Кейсы: lean (1), BIM (1)",
            "caseCount": 2,
        }
    )
    agent = AdvisorAgent(llm, _mock_chunk_store(hits))

    result = await agent.advise("Как сократить сроки?", role="employee", language="ru")

    assert isinstance(result, AdvisorResponse)
    assert result.refusal is False
    assert result.title == "Практические шаги"
    assert len(result.points) == 2
    assert result.points[0].case_id == "case-001"
    assert result.points[0].metric == "12%"
    assert result.caseCount == 2
    assert result.cost_usd == 0.0031


@pytest.mark.asyncio
async def test_advisor_resolves_source_slug_from_title() -> None:
    # FIX-9: a source_detail whose title matches a retrieved chunk gets that
    # chunk's slug (so the UI can link it); an unmatched title stays slug-less.
    hits = [_hit(file_id="case-001", slug="lean-project")]  # chunk title "Lean Project"
    llm = _mock_llm(
        {
            "title": "T",
            "summary": "S",
            "points": [
                {"heading": "H", "body": "B", "metric": "12%", "tag": "T", "case_id": "case-001"}
            ],
            "sources_detail": [
                {"title": "Lean Project", "kind": "Факт", "role": "Определяющий", "quote": "q"},
                {"title": "Неизвестный источник", "kind": "Факт", "role": "Подтверждающий", "quote": "q2"},
            ],
            "source": "s",
            "caseCount": 1,
        }
    )
    agent = AdvisorAgent(llm, _mock_chunk_store(hits))

    result = await agent.advise("q?", language="ru")

    by_title = {s.title: s for s in result.sources_detail}
    assert by_title["Lean Project"].slug == "lean-project"
    assert by_title["Неизвестный источник"].slug == ""


@pytest.mark.asyncio
async def test_advisor_refuses_when_similarity_below_threshold() -> None:
    hits = [_hit(similarity=0.12)]
    llm = _mock_llm({"title": "should not run"})
    agent = AdvisorAgent(llm, _mock_chunk_store(hits))

    result = await agent.advise("quantum physics?", language="ru")

    assert result.refusal is True
    assert "материалов" in result.refusal_message.lower()
    llm.complete.assert_not_called()


@pytest.mark.asyncio
async def test_advisor_refuses_off_topic_llm_response() -> None:
    hits = [_hit()]
    llm = _mock_llm({"refusal": True, "refusal_message": "Нет релевантных кейсов."})
    agent = AdvisorAgent(llm, _mock_chunk_store(hits))

    result = await agent.advise("Как снизить риски?", language="ru")

    assert result.refusal is True
    assert result.refusal_message == "Нет релевантных кейсов."


@pytest.mark.asyncio
async def test_advisor_strips_unverified_metrics() -> None:
    hits = [_hit(text="Команда внедрила чек-листы без цифр.")]
    llm = _mock_llm(
        {
            "title": "Контроль качества",
            "summary": "Чек-листы улучшили процесс.",
            "points": [
                {
                    "heading": "Чек-листы",
                    "body": "Цифровые чек-листы на объекте.",
                    "metric": "-18% брака",
                    "tag": "Качество",
                    "case_id": "case-001",
                }
            ],
            "source": "Кейс контроля качества",
            "caseCount": 1,
        }
    )
    agent = AdvisorAgent(llm, _mock_chunk_store(hits))

    result = await agent.advise("Как улучшить качество?", language="ru")

    assert result.refusal is False
    assert result.points[0].metric == ""


@pytest.mark.asyncio
async def test_advisor_rejects_hallucinated_case_id() -> None:
    hits = [_hit(file_id="case-real")]
    llm = _mock_llm(
        {
            "title": "Test",
            "summary": "Summary",
            "points": [
                {
                    "heading": "Bad",
                    "body": "Body",
                    "metric": "",
                    "tag": "Tag",
                    "case_id": "case-fake",
                }
            ],
            "source": "src",
            "caseCount": 1,
        }
    )
    agent = AdvisorAgent(llm, _mock_chunk_store(hits))

    result = await agent.advise("question?", language="en")

    assert result.refusal is True


@pytest.mark.asyncio
async def test_advisor_uses_custom_system_prompt() -> None:
    hits = [_hit()]
    llm = _mock_llm(
        {
            "title": "Legal focus",
            "summary": "Summary",
            "points": [
                {
                    "heading": "Risk",
                    "body": "Body",
                    "metric": "",
                    "tag": "Legal",
                    "case_id": "case-001",
                }
            ],
            "source": "src",
            "caseCount": 1,
        }
    )
    agent = AdvisorAgent(llm, _mock_chunk_store(hits))
    custom = "You are a legal advisor. Return only JSON."

    await agent.advise("question?", system_prompt=custom)

    llm.complete.assert_awaited_once()
    assert llm.complete.call_args.kwargs["system"] == custom
