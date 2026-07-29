"""Unit tests for studio artifact generation (mock the LLM + source gathering)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from llm_wiki.agents import artifacts as art
from llm_wiki.agents.artifacts import ArtifactError, _render_infographic_svg, generate_content


class _StubLLM:
    """Duck-typed LLMClient: load_prompt is a no-op, complete returns canned JSON."""

    def __init__(self, response: str) -> None:
        self._response = response

    def load_prompt(self, _name: str, **_kw: Any) -> str:
        return "prompt"

    async def complete(self, **_kw: Any) -> tuple[str, None]:
        return self._response, None


async def _gen(kind: str, response: str, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setattr(art, "_title_and_slugs", AsyncMock(return_value=("Title", ["slug"])))
    monkeypatch.setattr(art, "_load_bodies", lambda _slugs: ("source material", ["Страница-источник"]))
    return await generate_content(object(), _StubLLM(response), kind=kind, document_id="d", language="ru")


async def test_report_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    resp = json.dumps({
        "summary": "Кратко о материале.",
        "key_insight": "Главный вывод.",
        "risks": ["Риск один", "Риск два"],
        "recommendations": ["Шаг — пояснение"],
        "relevance_pct": 93,
        "citation_coverage_pct": 95,
        "effect_horizon": "6–12 мес",
        "source_language": "RU",
    })
    out = await _gen("report", resp, monkeypatch)
    assert out["summary"] == "Кратко о материале."
    assert out["key_insight"] == "Главный вывод."
    assert out["risks"] == ["Риск один", "Риск два"]
    assert out["effect_horizon"] == "6–12 мес"
    # programmatic fields: real source titles + computed reading time
    assert out["sources"] == ["Страница-источник"]
    assert out["reading_minutes"] >= 1


async def test_report_clamps_out_of_range_pct(monkeypatch: pytest.MonkeyPatch) -> None:
    resp = json.dumps({
        "summary": "s", "key_insight": "k", "risks": [], "recommendations": [],
        "relevance_pct": 250, "citation_coverage_pct": -10,
        "effect_horizon": "1–3 мес", "source_language": "EN",
    })
    out = await _gen("report", resp, monkeypatch)
    assert out["relevance_pct"] == 100
    assert out["citation_coverage_pct"] == 0


async def test_test_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    resp = json.dumps({"questions": [
        {"prompt": "q", "options": ["a", "b"], "correct": 1, "explanation": "e"},
    ]})
    out = await _gen("test", resp, monkeypatch)
    assert out["questions"][0]["correct"] == 1
    assert len(out["questions"][0]["options"]) == 2


async def test_cards_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    resp = json.dumps({
        "insight": "Главный вывод.",
        "context": "Что за материал.",
        "steps": [{"title": "Шаг A", "text": "пояснение"}],
        "risk": "Что сломается.",
        "action": "Собрать команду на час.",
        "action_minutes": 60,
        "relevance_pct": 93,
        "source_language": "RU",
    })
    out = await _gen("card", resp, monkeypatch)
    assert out["insight"] == "Главный вывод."
    assert out["steps"][0]["title"] == "Шаг A"
    assert out["action_minutes"] == 60


async def test_cards_clamps_badges(monkeypatch: pytest.MonkeyPatch) -> None:
    resp = json.dumps({
        "insight": "i", "context": "c",
        "steps": [{"title": f"s{n}", "text": "t"} for n in range(9)],  # capped at 6
        "risk": "r", "action": "a",
        "action_minutes": 100000, "relevance_pct": -5, "source_language": "EN",
    })
    out = await _gen("card", resp, monkeypatch)
    assert out["relevance_pct"] == 0
    assert out["action_minutes"] == 480
    assert len(out["steps"]) == 6


async def test_presentation_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    resp = json.dumps({"title": "Deck", "slides": [
        {"heading": "H", "bullets": ["one", "two"], "notes": "n"},
    ]})
    out = await _gen("presentation", resp, monkeypatch)
    assert out["title"] == "Deck"
    assert out["slides"][0]["bullets"] == ["one", "two"]


async def test_infographic_returns_svg(monkeypatch: pytest.MonkeyPatch) -> None:
    resp = json.dumps({
        "title": "Заголовок",
        "stats": [{"label": "Рел.", "value": "94%"}],
        "points": ["Пункт один", "Пункт два"],
    })
    out = await _gen("infographic", resp, monkeypatch)
    assert out["svg"].startswith("<svg")
    assert "94%" in out["svg"] and "Пункт один" in out["svg"]


async def test_no_source_content_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(art, "_title_and_slugs", AsyncMock(return_value=("T", [])))
    monkeypatch.setattr(art, "_load_bodies", lambda _slugs: ("", []))
    with pytest.raises(ArtifactError, match="No source content"):
        await generate_content(object(), _StubLLM("{}"), kind="report", document_id="d", language="ru")


def test_page_title_extraction() -> None:
    assert art._page_title("my-slug", "# Заголовок страницы\n\nтело") == "Заголовок страницы"
    assert art._page_title("my-slug", "просто текст без заголовка") == "my slug"


async def test_invalid_json_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ArtifactError, match="invalid JSON"):
        await _gen("report", "not json", monkeypatch)


async def test_unsupported_kind_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ArtifactError, match="Unsupported"):
        await _gen("podcast", "{}", monkeypatch)


def test_infographic_svg_escapes_and_caps() -> None:
    svg = _render_infographic_svg({
        "title": "<script>x</script>",
        "stats": [{"label": "a", "value": "1"}] * 5,  # capped at 3
        "points": ["p"] * 10,  # capped at 6
    })
    assert "<script>" not in svg  # escaped
    assert svg.count("<circle") == 6  # points capped
