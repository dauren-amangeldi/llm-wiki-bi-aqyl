"""Twins endpoints — persona roster and free-form multi-persona chat (BI-AQYL-TWINS)."""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import AsyncGenerator

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
    delete_twin_session,
    get_twin_persona,
    get_twin_session_messages,
    list_twin_personas,
    list_twin_presets,
    list_twin_sessions,
    set_twin_message_reactions,
    suggest_twin_personas,
    update_twin_session_personas,
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
        real_name=persona.real_name,
    )


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
                "color": p.color, "description": p.description,
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


# Демо-идентичности: их советы — общая витрина, видны всем и редактируемы
# всеми (локальный/демо-режим). Всё остальное — строго личное (Д3).
_DEMO_OWNERS = frozenset({"demo@bi.group", "demo", "anon"})


def _visible_to(caller: str, created_by: str | None) -> bool:
    return created_by is None or created_by in _DEMO_OWNERS or created_by == caller


async def _require_session_access(
    db: AsyncSession, session_id: str, caller: str
) -> None:
    """404 незнакомую/чужую сессию (не 403 — не палим существование чужих)."""
    row = await db.get(TwinSession, session_id)
    if row is None or not _visible_to(caller, row.created_by):
        raise HTTPException(status_code=404, detail="twin session not found")


@router.get("/twin/sessions")
async def get_twin_sessions(
    case_id: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    caller: str = Depends(get_user_key),
) -> list[dict[str, object]]:
    """Past councils — ONLY the caller's own (demo ones stay shared).

    BUG-16 (N+1): без ``case_id`` отдаёт ВСЕ видимые сессии одним запросом
    (в ответе есть ``case_id``) — дашборд раньше слал по запросу на каждый
    кейс, до сотен параллельных вызовов при каждом монтировании.
    """
    if case_id is None:
        from sqlalchemy import select as sa_select

        from llm_wiki.storage.metadata import TwinSession

        rows = (
            await db.scalars(
                sa_select(TwinSession).order_by(TwinSession.created_at.desc())
            )
        ).all()
        return [
            {
                "id": s.id,
                "case_id": s.case_id,
                "persona_ids": s.persona_ids,
                "created_at": s.created_at.isoformat(),
            }
            for s in rows
            if _visible_to(caller, s.created_by)
        ]
    return [
        {
            "id": s.id,
            "case_id": case_id,
            "persona_ids": s.persona_ids,
            "created_at": s.created_at.isoformat(),
        }
        for s in await list_twin_sessions(db, case_id)
        if _visible_to(caller, s.created_by)
    ]


@router.get("/twin/sessions/{session_id}/messages")
async def get_twin_session_transcript(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    caller: str = Depends(get_user_key),
) -> list[dict[str, object]]:
    """Return a session's persisted transcript (user / persona / verdict rows in
    order), so reopening a past council reloads the full conversation."""
    await _require_session_access(db, session_id, caller)
    return [
        {
            "role": m.role,
            "persona_id": m.persona_id,
            "seq": m.seq,
            "content": m.content,
            "created_at": m.created_at.isoformat() if m.created_at else "",
        }
        for m in await get_twin_session_messages(db, session_id)
    ]


@router.delete("/twin/sessions/{session_id}")
async def delete_twin_session_endpoint(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    caller: str = Depends(get_user_key),
) -> dict[str, bool]:
    """Delete a council session and its whole transcript (owner/demo only)."""
    await _require_session_access(db, session_id, caller)
    if not await delete_twin_session(db, session_id):
        raise HTTPException(status_code=404, detail="twin session not found")
    return {"ok": True}


class SessionPersonasRequest(BaseModel):
    persona_ids: list[str] = Field(min_length=1, max_length=3)


@router.patch("/twin/sessions/{session_id}/personas")
async def patch_twin_session_personas(
    session_id: str,
    body: SessionPersonasRequest,
    db: AsyncSession = Depends(get_db),
    caller: str = Depends(get_user_key),
) -> dict[str, bool]:
    """Change the council line-up of an existing session («Изменить состав»)."""
    await _require_session_access(db, session_id, caller)
    for pid in body.persona_ids:
        if await get_twin_persona(db, pid) is None:
            raise HTTPException(status_code=404, detail=f"Unknown persona: {pid}")
    if not await update_twin_session_personas(db, session_id, body.persona_ids):
        raise HTTPException(status_code=404, detail="twin session not found")
    return {"ok": True}


class ReactionsRequest(BaseModel):
    reactions: list[str] = Field(max_length=8)


@router.patch("/twin/sessions/{session_id}/messages/{seq}/reactions")
async def patch_twin_message_reactions(
    session_id: str,
    seq: int,
    body: ReactionsRequest,
    db: AsyncSession = Depends(get_db),
    caller: str = Depends(get_user_key),
) -> dict[str, bool]:
    """Persist the user's emoji reactions on one message of the transcript."""
    await _require_session_access(db, session_id, caller)
    if not await set_twin_message_reactions(db, session_id, seq, body.reactions):
        raise HTTPException(status_code=404, detail="twin message not found")
    return {"ok": True}


class TwinChatRequest(BaseModel):
    session_id: str | None = None
    case_id: str
    persona_ids: list[str] = Field(min_length=1, max_length=3)
    message: str = Field(default="", max_length=4000)
    language: str = "ru"
    # Opening round (right after «Начать совет»): no user message — every
    # persona briefly introduces their take on the case, messenger-style.
    opening: bool = False


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
    generate each persona's short messenger-style reply in sequence (each sees
    the ones just generated this turn), stream them, persist them.

    A persona may hand off to a colleague (``ask``) — that colleague replies in
    the same turn, capped at 2 extra replies so a debate can't ping-pong
    forever. ``opening=true`` skips the user message and routing: every persona
    introduces their take on the case (the round right after «Начать совет»).
    """
    if not body.opening and not body.message.strip():
        raise HTTPException(400, "message is required unless opening=true")
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
        if body.case_id != session_row.case_id:
            raise HTTPException(400, "case_id does not match the session's case")
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
            transcript = build_chat_transcript(existing, real_name_by_id)

            if body.opening:
                # Opening round: no user message, no routing — everyone speaks.
                transcript += (
                    "\nМодератор: Совет по кейсу создан. Каждый участник кратко"
                    " представляет свою стартовую позицию по кейсу — 1-2 коротких"
                    " сообщения, как в групповом чате."
                )
                responder_ids = [p.id for p in personas]
            else:
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
            # Persona→persona handoff: a reply's `ask` queues that colleague's
            # answer in the same turn, capped so a debate can't loop forever.
            max_extra = 2
            extra_used = 0
            queue = [pid for pid in responder_ids if pid in personas_by_id]
            while queue:
                pid = queue.pop(0)
                persona = personas_by_id.get(pid)
                if persona is None:
                    continue
                yield _sse_line({"event": "typing", "persona_id": pid})
                try:
                    reply = await agent.respond_as_persona(
                        persona, personas, case_context, transcript, body.language
                    )
                    bubbles = reply.messages
                    cite = reply.cite
                    reply_to = reply.reply_to
                    ask = reply.ask
                except Exception as exc:  # noqa: BLE001
                    logger.warning("twins_persona_reply_failed", persona_id=pid, error=str(exc))
                    bubbles, cite, reply_to, ask = ["Не удалось получить ответ."], "", "", ""

                for i, bubble in enumerate(bubbles):
                    if i > 0:
                        # Second bubble of the same persona: show «печатает…»
                        # again and pause briefly, so the chat reads like a
                        # person typing two messages in a row, not a dump.
                        yield _sse_line({"event": "typing", "persona_id": pid})
                        await asyncio.sleep(random.uniform(0.9, 1.7))
                    content: dict[str, object] = {
                        "text": bubble,
                        # reply_to on the first bubble (drives the «· Ответ:» label),
                        # cite on the last one (the supporting reference).
                        "reply_to": reply_to if i == 0 else "",
                        "cite": cite if i == len(bubbles) - 1 else "",
                    }
                    row = await append_twin_message(
                        db, session_id=session_row.id, role="persona",
                        persona_id=pid, seq=seq, content=content,
                    )
                    yield _sse_line({
                        "event": "message", "persona_id": pid, "seq": seq, "content": content,
                        "at": row.created_at.isoformat() if row.created_at else "",
                    })
                    transcript += f"\n{real_name_by_id.get(pid, pid)}: {bubble}"
                    seq += 1
                # Небольшая пауза перед следующим участником — вдобавок к
                # естественной задержке генерации его реплики.
                if queue:
                    await asyncio.sleep(random.uniform(0.5, 1.1))

                if (
                    ask
                    and ask != pid
                    and ask in personas_by_id
                    and ask not in queue
                    and extra_used < max_extra
                ):
                    queue.append(ask)
                    extra_used += 1

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

