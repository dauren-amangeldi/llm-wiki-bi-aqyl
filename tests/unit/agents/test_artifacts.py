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
    monkeypatch.setattr(art, "_load_bodies", lambda _slugs: "source material")
    return await generate_content(object(), _StubLLM(response), kind=kind, document_id="d", language="ru")


async def test_report_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    resp = json.dumps({
        "executive_summary": "s",
        "metrics": [{"label": "Рел.", "value": "94%"}],
        "sections": [{"heading": "h", "body": "b"}],
    })
    out = await _gen("report", resp, monkeypatch)
    assert out["executive_summary"] == "s"
    assert out["metrics"][0]["value"] == "94%"
    assert out["sections"][0]["heading"] == "h"


async def test_test_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    resp = json.dumps({"questions": [
        {"prompt": "q", "options": ["a", "b"], "correct": 1, "explanation": "e"},
    ]})
    out = await _gen("test", resp, monkeypatch)
    assert out["questions"][0]["correct"] == 1
    assert len(out["questions"][0]["options"]) == 2


async def test_cards_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    resp = json.dumps({
        "title": "T", "summary": "s",
        "key_points": [{"label": "Инсайт", "text": "t"}],
        "recommendations": ["r"], "tags": ["a", "b"],
    })
    out = await _gen("card", resp, monkeypatch)
    assert out["key_points"][0]["label"] == "Инсайт"
    assert out["tags"] == ["a", "b"]


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
    monkeypatch.setattr(art, "_load_bodies", lambda _slugs: "")
    with pytest.raises(ArtifactError, match="No source content"):
        await generate_content(object(), _StubLLM("{}"), kind="report", document_id="d", language="ru")


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
