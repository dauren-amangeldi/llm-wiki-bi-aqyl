"""Unit tests for the case auto-tagger classifier (item B, part 2)."""

from __future__ import annotations

import json
from typing import Any

from llm_wiki.agents.tagger import classify_case_tags


class _StubLLM:
    """Duck-typed stand-in for LLMClient — just enough for classify_case_tags."""

    def __init__(self, response: str):
        self._response = response

    def load_prompt(self, _name: str, **_kw: Any) -> str:
        return "prompt"

    async def complete(self, **_kw: Any) -> tuple[str, None]:
        return self._response, None


async def test_classify_keeps_only_taxonomy_tags() -> None:
    # LLM returns a real tag, an unknown one, and another real tag — the unknown
    # must be dropped and the result ordered by the taxonomy. The description
    # rides along in the same call (single LLM round-trip per case).
    llm = _StubLLM(
        json.dumps(
            {
                "tags": ["Финансы", "BogusTag", "Качество"],
                "description": "Кейс о финансовой дисциплине.",
            }
        )
    )
    tags, description = await classify_case_tags("t", "c", llm)  # type: ignore[arg-type]
    assert tags == ["Качество", "Финансы"]
    assert description == "Кейс о финансовой дисциплине."


async def test_classify_returns_empty_on_malformed_output() -> None:
    # A non-JSON payload must not raise — auto-tagging is best-effort.
    tags, description = await classify_case_tags("t", "c", _StubLLM("not json"))  # type: ignore[arg-type]
    assert tags == []
    assert description == ""
