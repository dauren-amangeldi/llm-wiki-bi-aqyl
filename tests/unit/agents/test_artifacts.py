"""Unit tests for studio artifact generation (mock the LLM + source gathering)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from llm_wiki.agents import artifacts as art
from llm_wiki.agents.artifacts import ArtifactError, _render_infographic_svg, generate_content


class _StubLLM:
    """Duck-typed LLMClient: load_prompt is a no-op, complete returns canned JSON,
    generate_image returns a data URI (or raises when ``image_error`` is set)."""

    def __init__(self, response: str, *, image_error: bool = False) -> None:
        self._response = response
        self._image_error = image_error

    def load_prompt(self, _name: str, **_kw: Any) -> str:
        return "prompt"

    async def complete(self, **_kw: Any) -> tuple[str, None]:
        return self._response, None

    async def generate_image(self, _prompt: str) -> str:
        if self._image_error:
            raise RuntimeError("no image model")
        return "data:image/png;base64,ZmFrZQ=="


async def _gen(
    kind: str,
    response: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    image_error: bool = False,
) -> dict[str, Any]:
    monkeypatch.setattr(art, "_title_and_slugs", AsyncMock(return_value=("Title", ["slug"])))
    monkeypatch.setattr(art, "_load_bodies", lambda _slugs: ("source material", ["Страница-источник"]))
    return await generate_content(
        object(), _StubLLM(response, image_error=image_error),
        kind=kind, document_id="d", language="ru",
    )


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
        "relevance_pct": 250, "citation_coverage_pct": -10,  # мусор от LLM
        "effect_horizon": "1–3 мес", "source_language": "EN",
    })
    out = await _gen("report", resp, monkeypatch)
    # BUG-26: псевдометрики удалены — даже если LLM их прислал, в выдачу не идут.
    assert "relevance_pct" not in out
    assert "citation_coverage_pct" not in out


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
    assert "relevance_pct" not in out  # BUG-26
    assert out["action_minutes"] == 480
    assert len(out["steps"]) == 6


async def test_presentation_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    resp = json.dumps({"title": "Deck", "slides": [
        {"heading": "H", "bullets": ["one", "two"], "notes": "n"},
    ]})
    out = await _gen("presentation", resp, monkeypatch)
    assert out["title"] == "Deck"
    assert out["slides"][0]["bullets"] == ["one", "two"]


_INFOGRAPHIC_RESP = json.dumps({
    "eyebrow": "Управление рисками",
    "headline": "Зрелость управления рисками",
    "key_insight": "Уровень 3–4 даёт на 35% меньше убытков.",
    "stats": [
        {"label": "Экономия", "value": "20 млн $"},
        {"label": "Меньше убытков", "value": "35%"},
        {"label": "Уровни", "value": "4"},
    ],
    "implementation_path": ["Assess", "Map", "Control", "Optimize"],
    "relevance_pct": 94,
    "source_language": "RU",
})


async def test_infographic_returns_generated_image_and_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    out = await _gen("infographic", _INFOGRAPHIC_RESP, monkeypatch)
    # Primary path: a generated picture, not an SVG.
    assert out["image_url"].startswith("data:image/png;base64,")
    assert "svg" not in out
    # structured fields drive the HTML cards next to the picture
    assert out["eyebrow"] == "Управление рисками"
    assert out["headline"] == "Зрелость управления рисками"
    assert out["key_insight"].startswith("Уровень 3–4")
    assert [s["value"] for s in out["stats"]] == ["20 млн $", "35%", "4"]
    assert out["implementation_path"] == ["Assess", "Map", "Control", "Optimize"]
    assert "relevance_pct" not in out  # BUG-26
    assert out["sources_count"] == 1  # from the mocked _load_bodies
    assert out["source_line"] == "Страница-источник"


async def test_infographic_falls_back_to_svg_when_image_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    out = await _gen("infographic", _INFOGRAPHIC_RESP, monkeypatch, image_error=True)
    # Fallback path: the self-contained SVG still renders with the same fields.
    assert "image_url" not in out
    assert out["svg"].startswith("<svg")
    assert "ГЛАВНЫЙ ИНСАЙТ" in out["svg"]
    assert "94%" not in out["svg"]  # BUG-26: плитки «Релевантность» в SVG нет
    assert "УПРАВЛЕНИЕ РИСКАМИ" in out["svg"]  # eyebrow uppercased
    assert "relevance_pct" not in out
    assert out["sources_count"] == 1


async def test_no_materials_raises_human_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """Пустой кейс (нет материалов вовсе) — человеческая причина в исключении."""
    monkeypatch.setattr(art, "_title_and_slugs", AsyncMock(return_value=("T", [])))
    monkeypatch.setattr(art, "_load_bodies", lambda _slugs: ("", []))
    with pytest.raises(ArtifactError, match="нет материалов"):
        await generate_content(object(), _StubLLM("{}"), kind="report", document_id="d", language="ru")


async def test_materials_without_content_raises_processing_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Материалы есть, но без обработанного текста — другая причина (ещё в работе)."""
    monkeypatch.setattr(art, "_title_and_slugs", AsyncMock(return_value=("T", ["slug-1"])))
    monkeypatch.setattr(art, "_load_bodies", lambda _slugs: ("", []))
    with pytest.raises(ArtifactError, match="обрабатываются"):
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
