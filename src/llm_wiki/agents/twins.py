"""TwinsAgent — multi-persona council deliberation (BI-AQYL-TWINS).

Pure business logic: receives personas/case context, returns dataclasses.
No FastAPI, Celery, or direct file I/O beyond reading already-written wiki pages.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog

from llm_wiki.agents.base import BaseAgent
from llm_wiki.llm.client import LLMClient

if TYPE_CHECKING:
    from llm_wiki.storage.metadata import FileRecord

logger = structlog.get_logger(__name__)


def load_case_context(documents: "list[FileRecord]") -> str:
    """Assemble case context text from the wiki pages of all linked documents.

    Mirrors ``AnswerAgent._answer_from_slugs``'s slug-aggregation pattern —
    same truncation limits, same object store — but returns raw context text
    instead of running a Q&A completion, since Twins personas need the whole
    case as background, not an answer to a single question.
    """
    from llm_wiki.agents.answer import MAX_PAGE_CHARS, MAX_TOTAL_CONTEXT_CHARS
    from llm_wiki.storage.object_store import get_object_store, wiki_key

    slugs: list[str] = []
    for doc in documents:
        slugs.extend(list(doc.created_pages or []))
        slugs.extend(list(doc.updated_pages or []))
    slugs = list(dict.fromkeys(slugs))

    store = get_object_store()
    blocks: list[str] = []
    total = 0
    for slug in slugs:
        try:
            body = store.get_text(wiki_key(slug)) or ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("twins_load_page_failed", slug=slug, error=str(exc))
            body = ""
        if not body:
            continue
        truncated = body[:MAX_PAGE_CHARS]
        if total + len(truncated) > MAX_TOTAL_CONTEXT_CHARS:
            break
        blocks.append(f"### [[{slug}]]\n\n{truncated}")
        total += len(truncated)

    return "\n\n".join(blocks)


_POSITION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reframing": {"type": "string"},
        "text": {"type": "string"},
        "cite": {"type": "string"},
    },
    "required": ["reframing", "text", "cite"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class TwinPersonaData:
    """Plain persona data passed into TwinsAgent (decoupled from the ORM row)."""

    id: str
    lens: str
    system_prompt: str
    domain_weights: dict[str, float]


@dataclass(frozen=True)
class PositionResult:
    persona_id: str
    reframing: str
    text: str
    cite: str


class TwinsAgent(BaseAgent):
    """Orchestrates the Twins council: position, cross-exam, and verdict rounds."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("call run_position_round / run_cross_exam_round / run_verdict_round")

    async def run_position_round(
        self,
        personas: list[TwinPersonaData],
        case_context: str,
        language: str,
        file_id: str = "twins",
    ) -> list[PositionResult]:
        """Run Round 1 for every persona in parallel: reframing + position in one call."""

        async def _one(persona: TwinPersonaData) -> PositionResult:
            prompt = self._llm.load_prompt(
                "twins_position",
                language=language,
                persona_lens=persona.lens,
                case_context=case_context,
            )
            text, _usage = await self._llm.complete(
                prompt=prompt,
                system=persona.system_prompt,
                file_id=file_id,
                agent_type="twins",
                json_schema=_POSITION_SCHEMA,
                schema_name="twins_position",
            )
            data = json.loads(text)
            return PositionResult(
                persona_id=persona.id,
                reframing=data["reframing"],
                text=data["text"],
                cite=data["cite"],
            )

        return list(await asyncio.gather(*[_one(p) for p in personas]))
