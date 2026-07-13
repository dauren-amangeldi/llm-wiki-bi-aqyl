"""AI-советник — стратегическая консультация: старт и переходы состояния."""

import json

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.agents.consultation import run_discovery
from llm_wiki.api.deps import get_db, get_user_key
from llm_wiki.api.schemas import (
    ClarificationRequiredResponse,
    ConsultationStartRequest,
    UnderstandingSnapshot,
    UnderstandingSnapshotResponse,
)
from llm_wiki.api.v1 import router
from llm_wiki.storage.metadata import append_chat_message, create_advisor_session, set_advisor_session_state


def _blank_snapshot(query: str) -> UnderstandingSnapshot:
    """Заготовка снимка, когда discovery решил, что вопросы не нужны — LLM
    заполнит поля на шаге respond, здесь только решение как черновик."""
    return UnderstandingSnapshot(
        decision=query,
        desired_outcome="",
        horizon="",
        constraints=[],
        stakeholders=[],
        success_criteria=[],
        assumptions=[],
    )


@router.post("/advisor/consultations")
async def start_consultation(
    body: ConsultationStartRequest,
    db: AsyncSession = Depends(get_db),
    user_key: str = Depends(get_user_key),
):
    """Начать консультацию: классифицировать решение и либо задать вопросы,
    либо сразу перейти к снимку понимания."""
    from llm_wiki.llm.chunk_store import ChunkStore
    from llm_wiki.llm.client import LLMClient

    session = await create_advisor_session(db, user_key=user_key, title=body.query)
    await append_chat_message(
        db, user_key=user_key, scope_type="advisor", scope_id=session.id, role="user", text_body=body.query
    )

    llm = LLMClient()
    try:
        chunk_store = ChunkStore(llm_client=llm)
        discovery = await run_discovery(llm, chunk_store, query=body.query, role=body.role, language=body.language)
    finally:
        await llm.aclose()

    if discovery.sufficient_context:
        await set_advisor_session_state(db, session.id, "context_review")
        snapshot = _blank_snapshot(body.query)
        await append_chat_message(
            db, user_key=user_key, scope_type="advisor", scope_id=session.id,
            role="assistant", text_body=json.dumps({"kind": "understanding_snapshot", "snapshot": snapshot.model_dump()}),
        )
        return UnderstandingSnapshotResponse(session_id=session.id, snapshot=snapshot)

    await append_chat_message(
        db, user_key=user_key, scope_type="advisor", scope_id=session.id,
        role="assistant",
        text_body=json.dumps({
            "kind": "clarification_required",
            "decision_type": discovery.decision_type,
            "questions": [q.model_dump() for q in discovery.questions],
        }),
    )
    return ClarificationRequiredResponse(
        session_id=session.id,
        decision_type=discovery.decision_type,
        questions=discovery.questions,
    )
