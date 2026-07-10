"""Twins council endpoints — persona roster and SSE deliberation (BI-AQYL-TWINS)."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Literal

import structlog
from fastapi import Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.agents.twins import TwinPersonaData, TwinsAgent, build_chat_transcript, load_case_context
from llm_wiki.api.deps import get_db, get_user_key
from llm_wiki.api.v1 import router
from llm_wiki.storage.metadata import (
    CaseRecord,
    FileRecord,
    TwinPersona,
    TwinSession,
    append_twin_message,
    create_twin_session,
    get_twin_persona,
    get_twin_session_messages,
    list_twin_personas,
    list_twin_presets,
    list_twin_sessions,
    set_twin_session_outcome,
    suggest_twin_personas,
)


logger = structlog.get_logger(__name__)


def _sse_line(payload: dict[str, object]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _to_persona_data(persona: TwinPersona) -> TwinPersonaData:
    return TwinPersonaData(
        id=persona.id,
        lens=persona.lens,
        system_prompt=persona.system_prompt,
        domain_weights=persona.domain_weights,
    )


class TwinCouncilRequest(BaseModel):
    case_id: str
    persona_ids: list[str] = Field(min_length=1, max_length=3)
    language: str = "ru"


@router.get("/twin/personas")
async def get_twin_roster(db: AsyncSession = Depends(get_db)) -> dict[str, object]:
    """Return the full persona roster and preset triads for the Twins UI."""
    personas = await list_twin_personas(db)
    presets = await list_twin_presets(db)
    return {
        "personas": [
            {
                "id": p.id, "name": p.name, "inspiration": p.inspiration,
                "real_name": p.real_name, "track": p.track,
                "pinned": bool(p.pinned), "lens": p.lens, "avatar_init": p.avatar_init,
            }
            for p in personas
        ],
        "presets": [{"id": p.id, "name": p.name, "persona_ids": p.persona_ids} for p in presets],
    }


@router.get("/twin/suggest")
async def suggest_council(
    case_id: str = Query(...), db: AsyncSession = Depends(get_db)
) -> dict[str, object]:
    """Suggest personas from the latest council of the most similar case.

    ``{"suggestion": null}`` when there is nothing to suggest — the UI falls
    back to the regular picker (same silent-empty convention as /similar).
    """
    return {"suggestion": await suggest_twin_personas(db, case_id)}


class OutcomeRequest(BaseModel):
    outcome: Literal["confirmed", "refuted"]
    note: str = Field(default="", max_length=1000)


@router.get("/twin/sessions")
async def get_twin_sessions(
    case_id: str = Query(...), db: AsyncSession = Depends(get_db)
) -> list[dict[str, object]]:
    """Past councils of a case with their outcome-journal state, newest first."""
    return [
        {
            "id": s.id,
            "persona_ids": s.persona_ids,
            "created_at": s.created_at.isoformat(),
            "outcome": s.outcome,
            "outcome_note": s.outcome_note,
        }
        for s in await list_twin_sessions(db, case_id)
    ]


@router.patch("/twin/sessions/{session_id}/outcome")
async def patch_twin_session_outcome(
    session_id: str, body: OutcomeRequest, db: AsyncSession = Depends(get_db)
) -> dict[str, bool]:
    """Record whether the council's verdict held up in reality."""
    if not await set_twin_session_outcome(db, session_id, body.outcome, body.note):
        raise HTTPException(status_code=404, detail="twin session not found")
    return {"ok": True}


class TwinChatRequest(BaseModel):
    session_id: str | None = None
    case_id: str
    persona_ids: list[str] = Field(min_length=1, max_length=3)
    message: str = Field(min_length=1, max_length=4000)
    language: str = "ru"


@router.post(
    "/twin/chat",
    summary="Twins chat — free-form multi-persona conversation with SSE streaming",
    tags=["twins"],
    response_class=StreamingResponse,
)
async def twin_chat_endpoint(
    body: TwinChatRequest,
    db: AsyncSession = Depends(get_db),
    user_key: str = Depends(get_user_key),
) -> StreamingResponse:
    """One chat turn: persist the user's message, route it to 0-3 personas,
    generate each persona's reply in sequence (each sees the ones just
    generated this turn), stream them, persist them.
    """
    personas_rows: list[TwinPersona] = []
    for persona_id in body.persona_ids:
        persona = await get_twin_persona(db, persona_id)
        if persona is None:
            raise HTTPException(404, f"Unknown persona: {persona_id}")
        personas_rows.append(persona)

    case = await db.get(CaseRecord, body.case_id)
    documents: list[FileRecord] = []
    if case is not None:
        for doc_id in case.doc_ids or []:
            fr = await db.get(FileRecord, doc_id)
            if fr is not None:
                documents.append(fr)

    if body.session_id:
        session_row = await db.get(TwinSession, body.session_id)
        if session_row is None:
            raise HTTPException(404, "Unknown session_id")
    else:
        session_row = await create_twin_session(
            db, case_id=body.case_id, persona_ids=body.persona_ids, created_by=user_key
        )

    personas = [_to_persona_data(p) for p in personas_rows]
    real_name_by_id = {p.id: p.real_name for p in personas_rows}
    case_context = load_case_context(documents)

    async def event_generator() -> AsyncGenerator[str, None]:
        from llm_wiki.llm.client import LLMClient

        llm = LLMClient()
        try:
            agent = TwinsAgent(llm)
            existing = await get_twin_session_messages(db, session_row.id)
            seq = (existing[-1].seq + 1) if existing else 0

            user_row = await append_twin_message(
                db, session_id=session_row.id, role="user", persona_id=None,
                seq=seq, content={"text": body.message},
            )
            seq += 1
            transcript = build_chat_transcript(existing + [user_row], real_name_by_id)

            try:
                responder_ids = await agent.route_message(personas, transcript, body.language)
            except Exception as exc:  # noqa: BLE001
                logger.warning("twins_route_failed_fallback_first_persona", error=str(exc))
                responder_ids = [personas[0].id]

            personas_by_id = {p.id: p for p in personas}
            for pid in responder_ids:
                persona = personas_by_id.get(pid)
                if persona is None:
                    continue
                yield _sse_line({"event": "typing", "persona_id": pid})
                try:
                    reply = await agent.respond_as_persona(persona, case_context, transcript, body.language)
                    content = {"text": reply.text, "cite": reply.cite}
                except Exception as exc:  # noqa: BLE001
                    logger.warning("twins_persona_reply_failed", persona_id=pid, error=str(exc))
                    content = {"text": "Не удалось получить ответ.", "cite": ""}
                await append_twin_message(
                    db, session_id=session_row.id, role="persona", persona_id=pid, seq=seq, content=content,
                )
                yield _sse_line({"event": "message", "persona_id": pid, "seq": seq, "content": content})
                transcript += f"\n{real_name_by_id.get(pid, pid)}: {content['text']}"
                seq += 1

            yield _sse_line({"done": True, "session_id": session_row.id})
        except Exception as exc:  # noqa: BLE001
            logger.exception("twins_chat_stream_failed", error=str(exc))
            yield _sse_line({"error": str(exc)})
        finally:
            await llm.aclose()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post(
    "/twin/council",
    summary="Twins council — multi-persona case deliberation with SSE streaming",
    tags=["twins"],
    response_class=StreamingResponse,
)
async def twin_council_endpoint(
    body: TwinCouncilRequest,
    stream: bool = Query(False),
    db: AsyncSession = Depends(get_db),
    user_key: str = Depends(get_user_key),
) -> StreamingResponse:
    """Run the Twins council (position → cross-exam → verdict) and stream SSE.

    Progress events use ``status``; per-persona replies use ``event: "message"``;
    the final event has ``done: true`` plus the ``session_id`` for later retrieval.
    """
    personas_rows: list[TwinPersona] = []
    for persona_id in body.persona_ids:
        persona = await get_twin_persona(db, persona_id)
        if persona is None:
            raise HTTPException(404, f"Unknown persona: {persona_id}")
        personas_rows.append(persona)

    case = await db.get(CaseRecord, body.case_id)
    documents: list[FileRecord] = []
    if case is not None:
        for doc_id in case.doc_ids or []:
            fr = await db.get(FileRecord, doc_id)
            if fr is not None:
                documents.append(fr)

    session_row = await create_twin_session(
        db, case_id=body.case_id, persona_ids=body.persona_ids, created_by=user_key
    )
    personas = [_to_persona_data(p) for p in personas_rows]
    case_context = load_case_context(documents)

    async def event_generator() -> AsyncGenerator[str, None]:
        from llm_wiki.llm.client import LLMClient

        llm = LLMClient()
        seq = 0
        try:
            agent = TwinsAgent(llm)

            yield _sse_line({"status": "position", "step": 1, "total": 3})
            positions = await agent.run_position_round(personas, case_context, body.language)
            for p in positions:
                content = {"reframing": p.reframing, "text": p.text, "cite": p.cite}
                await append_twin_message(
                    db, session_id=session_row.id, round="position", persona_id=p.persona_id,
                    seq=seq, content=content,
                )
                yield _sse_line({"event": "message", "round": "position", "persona_id": p.persona_id, "seq": seq, "content": content})
                seq += 1

            cross_exams: list = []
            if len(personas) >= 2:
                yield _sse_line({"status": "cross_exam", "step": 2, "total": 3})
                cross_exams = await agent.run_cross_exam_round(personas, case_context, positions, body.language)
                for c in cross_exams:
                    content = {
                        "disagreement": c.disagreement, "disagreement_forced": c.disagreement_forced,
                        "text": c.text, "cite": c.cite,
                    }
                    await append_twin_message(
                        db, session_id=session_row.id, round="cross_exam", persona_id=c.persona_id,
                        seq=seq, content=content,
                    )
                    yield _sse_line({"event": "message", "round": "cross_exam", "persona_id": c.persona_id, "seq": seq, "content": content})
                    seq += 1

            yield _sse_line({"status": "verdict", "step": 3, "total": 3})
            verdict = await agent.run_verdict_round(personas, case_context, positions, cross_exams, body.language)
            verdict_content = {
                "questions": verdict.questions, "consensus": verdict.consensus,
                "disagreement": verdict.disagreement, "next_step": verdict.next_step,
                "domain_distribution": verdict.domain_distribution,
                "decisive_voice": verdict.decisive_voice,
                "consensus_reached_early": verdict.consensus_reached_early,
                "is_close_split": verdict.is_close_split,
            }
            await append_twin_message(
                db, session_id=session_row.id, round="verdict", persona_id=None, seq=seq, content=verdict_content,
            )
            yield _sse_line({"event": "message", "round": "verdict", "persona_id": None, "seq": seq, "content": verdict_content})

            yield _sse_line({"done": True, "session_id": session_row.id})
        except Exception as exc:  # noqa: BLE001
            logger.exception("twins_council_stream_failed", error=str(exc))
            yield _sse_line({"error": str(exc)})
        finally:
            await llm.aclose()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
