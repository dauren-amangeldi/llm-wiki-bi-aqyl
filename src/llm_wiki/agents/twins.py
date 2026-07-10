"""TwinsAgent — multi-persona council deliberation (BI-AQYL-TWINS).

Pure business logic: receives personas/case context, returns dataclasses.
No FastAPI, Celery, or direct file I/O beyond reading already-written wiki pages.
"""

from __future__ import annotations

import asyncio
import difflib
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


_CROSS_EXAM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "disagreement": {"type": "string"},
        "text": {"type": "string"},
        "cite": {"type": "string"},
    },
    "required": ["disagreement", "text", "cite"],
    "additionalProperties": False,
}


def _is_novel_disagreement(
    candidate: str, prior_texts: list[str], threshold: float = 0.85
) -> bool:
    """True when *candidate* is not a near-duplicate of any text in *prior_texts*.

    Used to enforce the disagreement quota: a persona's Round 2 disagreement
    must not just restate its own Round 1 position.
    """
    candidate_norm = candidate.strip().lower()
    for prior in prior_texts:
        ratio = difflib.SequenceMatcher(None, candidate_norm, prior.strip().lower()).ratio()
        if ratio >= threshold:
            return False
    return True


@dataclass(frozen=True)
class CrossExamResult:
    persona_id: str
    disagreement: str
    disagreement_forced: bool
    text: str
    cite: str


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


def _build_transcript(positions: list[PositionResult], cross_exams: list[CrossExamResult]) -> str:
    lines = ["# Round 1 — Позиции"]
    for p in positions:
        lines.append(f"## {p.persona_id}\nПереформулировка: {p.reframing}\nПозиция: {p.text}")
    if cross_exams:
        lines.append("\n# Round 2 — Перекрёстный допрос")
        for c in cross_exams:
            lines.append(f"## {c.persona_id}\nРасхождение: {c.disagreement}\nРеплика: {c.text}")
    return "\n\n".join(lines)


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
    """Orchestrates the Twins council: position, cross-exam, and verdict rounds."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("call run_position_round / run_cross_exam_round / run_verdict_round")

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

        raw_results = await asyncio.gather(
            *[_one(p) for p in personas], return_exceptions=True
        )
        results: list[PositionResult] = []
        for persona, result in zip(personas, raw_results):
            if isinstance(result, Exception):
                logger.warning("twins_position_failed", persona_id=persona.id, error=str(result))
                continue
            results.append(result)
        return results

    async def run_cross_exam_round(
        self,
        personas: list[TwinPersonaData],
        case_context: str,
        positions: list[PositionResult],
        language: str,
        file_id: str = "twins",
    ) -> list[CrossExamResult]:
        """Run Round 2: each persona must surface a genuinely new disagreement.

        One retry is allowed if the model just restates its own Round 1 text.
        If the retry still fails the novelty check, the reply is kept but
        flagged ``disagreement_forced=True`` — visible to the caller instead
        of silently passing off manufactured conflict as organic.
        """
        positions_block = "\n\n".join(f"### {p.persona_id}\n{p.text}" for p in positions)
        own_text_by_id = {p.persona_id: p.text for p in positions}

        async def _one(persona: TwinPersonaData) -> CrossExamResult:
            prompt = self._llm.load_prompt(
                "twins_cross_exam",
                language=language,
                persona_lens=persona.lens,
                case_context=case_context,
                positions_block=positions_block,
            )
            own_text = own_text_by_id.get(persona.id, "")

            text, _usage = await self._llm.complete(
                prompt=prompt,
                system=persona.system_prompt,
                file_id=file_id,
                agent_type="twins",
                json_schema=_CROSS_EXAM_SCHEMA,
                schema_name="twins_cross_exam",
            )
            data = json.loads(text)
            if _is_novel_disagreement(data["disagreement"], [own_text]):
                return CrossExamResult(
                    persona_id=persona.id, disagreement=data["disagreement"],
                    disagreement_forced=False, text=data["text"], cite=data["cite"],
                )

            first_attempt = data
            try:
                text, _usage = await self._llm.complete(
                    prompt=prompt,
                    system=persona.system_prompt,
                    file_id=file_id,
                    agent_type="twins",
                    json_schema=_CROSS_EXAM_SCHEMA,
                    schema_name="twins_cross_exam",
                )
                data = json.loads(text)
            except Exception as exc:  # noqa: BLE001
                logger.warning("twins_cross_exam_retry_failed", persona_id=persona.id, error=str(exc))
                return CrossExamResult(
                    persona_id=persona.id, disagreement=first_attempt["disagreement"],
                    disagreement_forced=True, text=first_attempt["text"], cite=first_attempt["cite"],
                )

            is_novel = _is_novel_disagreement(data["disagreement"], [own_text])
            return CrossExamResult(
                persona_id=persona.id, disagreement=data["disagreement"],
                disagreement_forced=not is_novel, text=data["text"], cite=data["cite"],
            )

        raw_results = await asyncio.gather(
            *[_one(p) for p in personas], return_exceptions=True
        )
        results: list[CrossExamResult] = []
        for persona, result in zip(personas, raw_results):
            if isinstance(result, Exception):
                logger.warning("twins_cross_exam_failed", persona_id=persona.id, error=str(result))
                continue
            results.append(result)
        return results

    async def run_verdict_round(
        self,
        personas: list[TwinPersonaData],
        case_context: str,
        positions: list[PositionResult],
        cross_exams: list[CrossExamResult],
        language: str,
        file_id: str = "twins",
    ) -> VerdictResult:
        """Synthesize the final verdict and compute the domain-weighted decisive voice.

        ``decisive_voice`` is never invented by the LLM — it's derived
        deterministically from ``domain_distribution`` (LLM output) crossed
        with each persona's stored ``domain_weights``, so the result is
        auditable instead of a black box.
        """
        transcript = _build_transcript(positions, cross_exams)
        prompt = self._llm.load_prompt(
            "twins_verdict", language=language, case_context=case_context, transcript=transcript
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
        consensus_reached_early = bool(cross_exams) and all(c.disagreement_forced for c in cross_exams)

        return VerdictResult(
            questions=data["questions"],
            consensus=data["consensus"],
            disagreement=data["disagreement"],
            next_step=data["next_step"],
            domain_distribution=data["domain_distribution"],
            decisive_voice=decisive_id,
            consensus_reached_early=consensus_reached_early,
            is_close_split=is_close_split,
        )
