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


async def test_infographic_returns_svg_and_structured_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    resp = json.dumps({
        "eyebrow": "Управление рисками",
        "key_insight": "Уровень 3–4 даёт на 35% меньше убытков.",
        "implementation_path": ["Assess", "Map", "Control", "Optimize"],
        "relevance_pct": 94,
        "source_language": "RU",
    })
    out = await _gen("infographic", resp, monkeypatch)
    # SVG carries the visible text
    assert out["svg"].startswith("<svg")
    assert "ГЛАВНЫЙ ИНСАЙТ" in out["svg"] and "94%" in out["svg"]
    assert "Assess → Map → Control → Optimize" in out["svg"]
    assert "УПРАВЛЕНИЕ РИСКАМИ" in out["svg"]  # eyebrow uppercased
    # structured fields kept alongside for Copy / .md export
    assert out["key_insight"].startswith("Уровень 3–4")
    assert out["implementation_path"] == ["Assess", "Map", "Control", "Optimize"]
    assert out["relevance_pct"] == 94
    assert out["sources_count"] == 1  # from the mocked _load_bodies


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


def test_infographic_svg_escapes_and_clamps() -> None:
    svg = _render_infographic_svg(
        {
            "eyebrow": "<script>x</script>",
            "key_insight": "<b>bold?</b>",
            "implementation_path": [f"Step{n}" for n in range(9)],  # capped at 6
            "relevance_pct": 250,  # clamped to 100
            "source_language": "ru",
        },
        title="<img onerror=alert(1)>",
        source_titles=["Страница A", "Страница B"],
    )
    assert "<script>" not in svg and "<img" not in svg and "<b>" not in svg  # escaped
    assert "100%" in svg  # relevance clamped
    assert ">2<" in svg  # sources_count = len(source_titles)
    assert "Step5" in svg and "Step6" not in svg  # path capped at 6 (Step0..Step5)
