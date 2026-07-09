"""Twins council endpoints — persona roster and SSE deliberation (BI-AQYL-TWINS)."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator

from fastapi import Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.agents.twins import TwinPersonaData, TwinsAgent, load_case_context
from llm_wiki.api.deps import get_db, get_user_key
from llm_wiki.api.v1 import router
from llm_wiki.storage.metadata import (
    CaseRecord,
    FileRecord,
    TwinPersona,
    append_twin_message,
    create_twin_session,
    get_twin_persona,
    list_twin_personas,
    list_twin_presets,
)


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
                "id": p.id, "name": p.name, "inspiration": p.inspiration, "track": p.track,
                "pinned": bool(p.pinned), "lens": p.lens, "avatar_init": p.avatar_init,
            }
            for p in personas
        ],
        "presets": [{"id": p.id, "name": p.name, "persona_ids": p.persona_ids} for p in presets],
    }


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
            }
            await append_twin_message(
                db, session_id=session_row.id, round="verdict", persona_id=None, seq=seq, content=verdict_content,
            )
            yield _sse_line({"event": "message", "round": "verdict", "persona_id": None, "seq": seq, "content": verdict_content})

            yield _sse_line({"done": True, "session_id": session_row.id})
        except Exception as exc:  # noqa: BLE001
            yield _sse_line({"error": str(exc)})
        finally:
            await llm.aclose()

    return StreamingResponse(event_generator(), media_type="text/event-stream")
