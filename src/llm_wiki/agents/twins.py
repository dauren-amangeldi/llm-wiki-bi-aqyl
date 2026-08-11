"""TwinsAgent — free-form multi-persona chat (BI-AQYL-TWINS).

Pure business logic: receives personas/case context, returns dataclasses.
No FastAPI, Celery, or direct file I/O beyond reading already-written wiki pages.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog

from llm_wiki.agents.base import BaseAgent
from llm_wiki.llm.client import LLMClient

if TYPE_CHECKING:
    from llm_wiki.storage.metadata import FileRecord, TwinMessage

logger = structlog.get_logger(__name__)


def load_case_context(documents: "list[FileRecord]") -> str:
    """Assemble case context text from the wiki pages of all linked documents.

    Mirrors ``AnswerAgent._answer_from_slugs``'s slug-aggregation pattern —
    same truncation limits, same object store — but returns raw context text
    instead of running a Q&A completion, since Twins personas need the whole
    case as background, not an answer to a single question.
    """
    from llm_wiki.agents.answer import MAX_PAGE_CHARS, MAX_TOTAL_CONTEXT_CHARS
    from llm_wiki.storage import wiki_store

    slugs: list[str] = []
    for doc in documents:
        slugs.extend(list(doc.created_pages or []))
        slugs.extend(list(doc.updated_pages or []))
    slugs = list(dict.fromkeys(slugs))

    # Wiki page bodies live in Postgres (wiki_fts) now, not the object store —
    # same source AnswerAgent reads. caller=None: trusted internal read (the
    # session is already scoped to a case the user owns at the endpoint layer).
    blocks: list[str] = []
    total = 0
    for slug in slugs:
        try:
            body = wiki_store.get_page(slug) or ""
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


def build_chat_transcript(
    messages: "list[TwinMessage]", real_name_by_id: dict[str, str]
) -> str:
    """Render persisted TwinMessage rows as a plain-text transcript for prompts."""
    lines: list[str] = []
    for m in messages:
        if m.role == "user":
            speaker = "Пользователь"
        elif m.role == "verdict":
            speaker = "Итог"
        else:
            speaker = real_name_by_id.get(m.persona_id or "", m.persona_id or "?")
        text = m.content.get("text", "") if isinstance(m.content, dict) else ""
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


@dataclass(frozen=True)
class TwinPersonaData:
    """Plain persona data passed into TwinsAgent (decoupled from the ORM row)."""

    id: str
    lens: str
    system_prompt: str
    domain_weights: dict[str, float]
    real_name: str = ""


def _chat_reply_schema(other_persona_ids: list[str]) -> dict[str, Any]:
    """Reply schema: 1-2 short messenger-style bubbles + optional handoff.

    ``ask`` lets the persona address a colleague (empty string = nobody) — the
    endpoint then generates that colleague's reply in the same turn (capped).
    """
    return {
        "type": "object",
        "properties": {
            "messages": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 2,
            },
            "cite": {"type": "string"},
            "reply_to": {"type": "string"},
            "ask": {"type": "string", "enum": ["", *other_persona_ids]},
        },
        "required": ["messages", "cite", "reply_to", "ask"],
        "additionalProperties": False,
    }

_ROUTER_SYSTEM_PROMPT = "Ты нейтральный маршрутизатор AI-совета Twins. Не отвечай от лица персон."


def _route_schema(persona_ids: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "responders": {
                "type": "array",
                "items": {"type": "string", "enum": persona_ids},
                "maxItems": 3,
            },
        },
        "required": ["responders"],
        "additionalProperties": False,
    }


@dataclass(frozen=True)
class ChatReplyResult:
    persona_id: str
    messages: list[str]
    cite: str
    reply_to: str
    ask: str


def _participants_block(personas: list[TwinPersonaData]) -> str:
    return "\n".join(f"- {p.real_name or p.id} ({p.id}) — {p.lens}" for p in personas)


class TwinsAgent(BaseAgent):
    """Routes chat turns to personas and generates each persona's short
    messenger-style reply in sequence (personas can hand off to each other)."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("call route_message / respond_as_persona")

    async def route_message(
        self,
        personas: list[TwinPersonaData],
        chat_transcript: str,
        language: str,
        file_id: str = "twins",
    ) -> list[str]:
        """Decide which 0-3 personas should respond to the latest message, in order."""
        persona_ids = [p.id for p in personas]
        personas_block = "\n".join(
            f"- {p.id}: {p.real_name or p.id} — {p.lens}" for p in personas
        )
        prompt = self._llm.load_prompt(
            "twins_route", language=language, personas_block=personas_block,
            chat_transcript=chat_transcript,
        )
        text, _usage = await self._llm.complete(
            prompt=prompt,
            system=_ROUTER_SYSTEM_PROMPT,
            file_id=file_id,
            agent_type="twins",
            json_schema=_route_schema(persona_ids),
            schema_name="twins_route",
        )
        data = json.loads(text)
        known = set(persona_ids)
        return [pid for pid in data["responders"] if pid in known]

    async def respond_as_persona(
        self,
        persona: TwinPersonaData,
        participants: list[TwinPersonaData],
        case_context: str,
        chat_transcript: str,
        language: str,
        file_id: str = "twins",
    ) -> ChatReplyResult:
        """Generate one persona's short reply, seeing any replies already added
        this turn. The persona knows who else is in the chat and may address a
        colleague via ``ask`` (the endpoint continues the chain, capped)."""
        others = [p for p in participants if p.id != persona.id]
        prompt = self._llm.load_prompt(
            "twins_chat_reply", language=language, persona_lens=persona.lens,
            persona_name=persona.real_name or persona.id,
            participants_block=_participants_block(participants),
            case_context=case_context, chat_transcript=chat_transcript,
        )
        text, _usage = await self._llm.complete(
            prompt=prompt,
            system=persona.system_prompt,
            file_id=file_id,
            agent_type="twins",
            json_schema=_chat_reply_schema([p.id for p in others]),
            schema_name="twins_chat_reply",
        )
        data = json.loads(text)
        messages = [str(m).strip() for m in data["messages"] if str(m).strip()]
        return ChatReplyResult(
            persona_id=persona.id,
            messages=messages or ["…"],
            cite=str(data.get("cite", "")),
            reply_to=str(data.get("reply_to", "")),
            ask=str(data.get("ask", "")),
        )

