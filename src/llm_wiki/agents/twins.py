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


_VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "persona_id": {"type": "string"},
                },
                "required": ["text", "persona_id"],
                "additionalProperties": False,
            },
        },
        "consensus": {"type": "string"},
        "disagreement": {"type": "string"},
        "next_step": {"type": "string"},
        "domain_distribution": {
            "type": "object",
            "properties": {
                "tech": {"type": "number"},
                "real_estate": {"type": "number"},
                "finance": {"type": "number"},
            },
            "required": ["tech", "real_estate", "finance"],
            "additionalProperties": False,
        },
    },
    "required": ["questions", "consensus", "disagreement", "next_step", "domain_distribution"],
    "additionalProperties": False,
}

_VERDICT_SYSTEM_PROMPT = "Ты синтезируешь честный вердикт совета AI-персон, не сглаживая разногласия."


def _compute_decisive_voice(
    persona_ids: list[str],
    domain_weights: dict[str, dict[str, float]],
    domain_distribution: dict[str, float],
) -> tuple[str, bool]:
    """Return (decisive persona_id, is_close_split) from weights × domain shares.

    ``is_close_split`` is True when the top two scores differ by less than
    0.15 — surfaced to the caller as an honest split instead of a false tie-break.
    """
    scores = {
        pid: sum(
            domain_weights.get(pid, {}).get(domain, 0.0) * share
            for domain, share in domain_distribution.items()
        )
        for pid in persona_ids
    }
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    decisive_id, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    is_close_split = (top_score - second_score) < 0.15
    return decisive_id, is_close_split


@dataclass(frozen=True)
class VerdictResult:
    questions: list[dict[str, str]]
    consensus: str
    disagreement: str
    next_step: str
    domain_distribution: dict[str, float]
    decisive_voice: str
    consensus_reached_early: bool
    is_close_split: bool


_CHAT_REPLY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "cite": {"type": "string"},
    },
    "required": ["text", "cite"],
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
    text: str
    cite: str


class TwinsAgent(BaseAgent):
    """Routes chat turns to personas, generates each persona's reply in sequence,
    and produces an on-demand verdict summarizing the chat so far."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("call route_message / respond_as_persona / run_chat_verdict")

    async def route_message(
        self,
        personas: list[TwinPersonaData],
        chat_transcript: str,
        language: str,
        file_id: str = "twins",
    ) -> list[str]:
        """Decide which 0-3 personas should respond to the latest message, in order."""
        persona_ids = [p.id for p in personas]
        personas_block = "\n".join(f"- {p.id}: {p.lens}" for p in personas)
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
        case_context: str,
        chat_transcript: str,
        language: str,
        file_id: str = "twins",
    ) -> ChatReplyResult:
        """Generate one persona's reply, seeing any replies already added this turn."""
        prompt = self._llm.load_prompt(
            "twins_chat_reply", language=language, persona_lens=persona.lens,
            case_context=case_context, chat_transcript=chat_transcript,
        )
        text, _usage = await self._llm.complete(
            prompt=prompt,
            system=persona.system_prompt,
            file_id=file_id,
            agent_type="twins",
            json_schema=_CHAT_REPLY_SCHEMA,
            schema_name="twins_chat_reply",
        )
        data = json.loads(text)
        return ChatReplyResult(persona_id=persona.id, text=data["text"], cite=data["cite"])

    async def run_chat_verdict(
        self,
        personas: list[TwinPersonaData],
        case_context: str,
        chat_transcript: str,
        language: str,
        file_id: str = "twins",
    ) -> VerdictResult:
        """Summarize a free-form chat on demand ("Подвести итог").

        Reuses the same schema/prompt as the old scripted verdict round —
        it only ever needed `case_context` + a transcript string, so a
        chat-formatted transcript works unchanged. `consensus_reached_early`
        doesn't map onto free-form chat (that flag came from the old
        cross-exam forced-disagreement heuristic) — always False here.
        """
        prompt = self._llm.load_prompt(
            "twins_verdict", language=language, case_context=case_context,
            transcript=chat_transcript,
        )
        text, _usage = await self._llm.complete(
            prompt=prompt,
            system=_VERDICT_SYSTEM_PROMPT,
            file_id=file_id,
            agent_type="twins",
            json_schema=_VERDICT_SCHEMA,
            schema_name="twins_verdict",
        )
        data = json.loads(text)

        domain_weights = {p.id: p.domain_weights for p in personas}
        decisive_id, is_close_split = _compute_decisive_voice(
            [p.id for p in personas], domain_weights, data["domain_distribution"]
        )
        return VerdictResult(
            questions=data["questions"],
            consensus=data["consensus"],
            disagreement=data["disagreement"],
            next_step=data["next_step"],
            domain_distribution=data["domain_distribution"],
            decisive_voice=decisive_id,
            consensus_reached_early=False,
            is_close_split=is_close_split,
        )

