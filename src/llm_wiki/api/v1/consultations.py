"""AI-советник — стратегическая консультация: старт и переходы состояния."""

import json

from fastapi import Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.agents.consultation import build_snapshot, run_discovery
from llm_wiki.api.deps import get_db, get_user_key
from llm_wiki.api.schemas import (
    ClarificationQuestion,
    ClarificationRequiredResponse,
    ConsultationRespondRequest,
    ConsultationSnapshotUpdate,
    ConsultationStartRequest,
    UnderstandingSnapshot,
    UnderstandingSnapshotResponse,
)
from llm_wiki.api.v1 import router
from llm_wiki.storage.metadata import (
    append_chat_message,
    create_advisor_session,
    get_advisor_session,
    list_chat_messages,
    set_advisor_session_state,
)


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


def _latest_snapshot(history) -> UnderstandingSnapshot | None:
    """Найти последний snapshot в истории сообщений."""
    for m in reversed(history):
        if m.role != "assistant":
            continue
        try:
            payload = json.loads(m.text)
        except (json.JSONDecodeError, TypeError):
            continue
        if payload.get("kind") == "understanding_snapshot":
            return UnderstandingSnapshot(**payload["snapshot"])
    return None


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


@router.post("/advisor/consultations/{session_id}/respond")
async def respond_to_questions(
    session_id: str,
    body: ConsultationRespondRequest,
    db: AsyncSession = Depends(get_db),
    user_key: str = Depends(get_user_key),
):
    """Принять ответы на блок вопросов (или give_advice_now) и построить снимок понимания."""
    session = await get_advisor_session(db, session_id)
    if session is None or session.user_key != user_key:
        raise HTTPException(status_code=404, detail="Advisor session not found")

    history = await list_chat_messages(db, user_key=user_key, scope_type="advisor", scope_id=session_id)
    original_query = next((m.text for m in history if m.role == "user"), "")

    questions: list[ClarificationQuestion] = []
    for m in history:
        if m.role != "assistant":
            continue
        try:
            payload = json.loads(m.text)
        except (json.JSONDecodeError, TypeError):
            continue
        if payload.get("kind") == "clarification_required":
            questions = [ClarificationQuestion(**q) for q in payload.get("questions", [])]

    from llm_wiki.llm.client import LLMClient

    llm = LLMClient()
    try:
        snapshot = await build_snapshot(llm, query=original_query, questions=questions, answers=body.answers)
    finally:
        await llm.aclose()

    await set_advisor_session_state(db, session_id, "context_review")
    await append_chat_message(
        db, user_key=user_key, scope_type="advisor", scope_id=session_id,
        role="assistant", text_body=json.dumps({"kind": "understanding_snapshot", "snapshot": snapshot.model_dump()}),
    )
    return UnderstandingSnapshotResponse(session_id=session_id, snapshot=snapshot)


@router.put("/advisor/consultations/{session_id}/snapshot")
async def update_snapshot(
    session_id: str,
    body: ConsultationSnapshotUpdate,
    db: AsyncSession = Depends(get_db),
    user_key: str = Depends(get_user_key),
):
    """Ручная правка снимка понимания перед подтверждением — частичный апдейт."""
    session = await get_advisor_session(db, session_id)
    if session is None or session.user_key != user_key:
        raise HTTPException(status_code=404, detail="Advisor session not found")

    history = await list_chat_messages(db, user_key=user_key, scope_type="advisor", scope_id=session_id)
    current = _latest_snapshot(history)
    if current is None:
        raise HTTPException(status_code=409, detail="No snapshot to update yet")

    updated = current.model_copy(update={k: v for k, v in body.model_dump().items() if v is not None})
    await append_chat_message(
        db, user_key=user_key, scope_type="advisor", scope_id=session_id,
        role="assistant", text_body=json.dumps({"kind": "understanding_snapshot", "snapshot": updated.model_dump()}),
    )
    return UnderstandingSnapshotResponse(session_id=session_id, snapshot=updated)
