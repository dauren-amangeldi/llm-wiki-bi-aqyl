"""Chat history endpoints — persisted per user + (document|case) scope."""

from fastapi import Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.api.deps import get_db, get_user_key
from llm_wiki.api.v1 import router
from llm_wiki.storage.metadata import (
    ChatRecord,
    clear_chat_messages,
    list_chat_messages,
)


class ChatCitation(BaseModel):
    """Citation anchor pointing to a wiki slug, with its human display title."""

    anchor: str
    title: str = ""


class ChatTurn(BaseModel):
    """One persisted chat turn, shaped to match the frontend ChatMessage."""

    role: str
    text: str
    citations: list[ChatCitation] = []
    model_name: str | None = None


def _anchor_title(anchor: str) -> str:
    """Resolve a stored citation anchor (a wiki slug) to its current page title
    so reloaded history shows real titles instead of raw slugs/UUIDs. Only the
    anchor is persisted, so the title is re-derived from the wiki on read."""
    from llm_wiki.storage import wiki_store

    try:
        title = wiki_store.get_page_title(anchor)
    except Exception:  # noqa: BLE001
        title = None
    return (title or "").strip() or anchor.replace("-", " ").title()


def _to_turn(record: ChatRecord) -> ChatTurn:
    return ChatTurn(
        role=record.role,
        text=record.text,
        citations=[
            ChatCitation(anchor=a, title=_anchor_title(a))
            for a in (record.citations or [])
        ],
        model_name=record.model_name,
    )


@router.get("/documents/{document_id}/chat", response_model=list[ChatTurn])
async def get_document_chat(
    document_id: str,
    user_key: str = Depends(get_user_key),
    db: AsyncSession = Depends(get_db),
) -> list[ChatTurn]:
    """Return the caller's chat history for a document, oldest-first."""
    rows = await list_chat_messages(
        db, user_key=user_key, scope_type="document", scope_id=document_id
    )
    return [_to_turn(r) for r in rows]


@router.delete("/documents/{document_id}/chat")
async def delete_document_chat(
    document_id: str,
    user_key: str = Depends(get_user_key),
    db: AsyncSession = Depends(get_db),
) -> dict[str, int]:
    """Clear the caller's chat history for a document."""
    n = await clear_chat_messages(
        db, user_key=user_key, scope_type="document", scope_id=document_id
    )
    return {"deleted": n}


@router.get("/cases/{case_id}/chat", response_model=list[ChatTurn])
async def get_case_chat(
    case_id: str,
    user_key: str = Depends(get_user_key),
    db: AsyncSession = Depends(get_db),
) -> list[ChatTurn]:
    """Return the caller's chat history for a case, oldest-first."""
    rows = await list_chat_messages(
        db, user_key=user_key, scope_type="case", scope_id=case_id
    )
    return [_to_turn(r) for r in rows]


@router.delete("/cases/{case_id}/chat")
async def delete_case_chat(
    case_id: str,
    user_key: str = Depends(get_user_key),
    db: AsyncSession = Depends(get_db),
) -> dict[str, int]:
    """Clear the caller's chat history for a case."""
    n = await clear_chat_messages(
        db, user_key=user_key, scope_type="case", scope_id=case_id
    )
    return {"deleted": n}
