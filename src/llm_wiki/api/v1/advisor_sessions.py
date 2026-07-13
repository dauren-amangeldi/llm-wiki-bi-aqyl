"""Advisor conversation history — list past sessions and resume one."""

from fastapi import Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.api.deps import get_db, get_user_key
from llm_wiki.api.v1 import router
from llm_wiki.storage.metadata import get_advisor_session, list_advisor_sessions, list_chat_messages


@router.get("/advisor/sessions")
async def list_sessions(
    db: AsyncSession = Depends(get_db),
    user_key: str = Depends(get_user_key),
) -> list[dict[str, object]]:
    """Return the caller's past advisor conversations, most recent first."""
    rows = await list_advisor_sessions(db, user_key=user_key)
    return [
        {
            "id": r.id,
            "title": r.title,
            "created_at": r.created_at.isoformat(),
            "updated_at": r.updated_at.isoformat(),
        }
        for r in rows
    ]


@router.get("/advisor/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    user_key: str = Depends(get_user_key),
) -> list[dict[str, object]]:
    """Return a session's messages, oldest first, for resuming it.

    404 both when the session doesn't exist and when it belongs to another
    user — same response either way, so a probing client can't tell which.
    """
    row = await get_advisor_session(db, session_id)
    if row is None or row.user_key != user_key:
        raise HTTPException(status_code=404, detail="Advisor session not found")
    messages = await list_chat_messages(
        db, user_key=user_key, scope_type="advisor", scope_id=session_id
    )
    return [
        {"role": m.role, "text": m.text, "created_at": m.created_at.isoformat()}
        for m in messages
    ]
