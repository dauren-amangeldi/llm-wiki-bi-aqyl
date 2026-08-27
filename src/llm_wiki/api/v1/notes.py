"""Личные заметки к материалам/кейсам (BUG-24).

«Мои заметки» жили только в localStorage браузера — терялись при смене
устройства/чистке и не были видны пользователю нигде, кроме одной машины.
Теперь сервер — источник истины; ключ (owner, doc_id), заметка приватна.
"""

from __future__ import annotations

from typing import Any

from fastapi import Depends
from pydantic import BaseModel, Field
from sqlalchemy import delete as sa_delete
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.api.deps import get_db, get_user_key
from llm_wiki.api.v1 import router
from llm_wiki.storage.metadata import MaterialNote


@router.get("/notes/{doc_id}")
async def get_note(
    doc_id: str,
    db: AsyncSession = Depends(get_db),
    caller: str = Depends(get_user_key),
) -> dict[str, Any]:
    """Заметка вызывающего к материалу. Нет строки — пустой текст (не 404:
    отсутствие заметки — нормальное состояние, а не ошибка)."""
    row = await db.get(MaterialNote, (caller, doc_id))
    return {
        "doc_id": doc_id,
        "text": row.text if row else "",
        "updated_at": row.updated_at.isoformat() if row and row.updated_at else None,
    }


class NoteBody(BaseModel):
    text: str = Field(max_length=50_000)


@router.put("/notes/{doc_id}")
async def put_note(
    doc_id: str,
    body: NoteBody,
    db: AsyncSession = Depends(get_db),
    caller: str = Depends(get_user_key),
) -> dict[str, Any]:
    """Upsert заметки; пустой текст удаляет строку (заметка «стёрта»)."""
    if not body.text.strip():
        await db.execute(
            sa_delete(MaterialNote).where(
                MaterialNote.owner == caller, MaterialNote.doc_id == doc_id
            )
        )
        await db.commit()
        return {"ok": True, "doc_id": doc_id, "text": ""}

    stmt = pg_insert(MaterialNote).values(
        owner=caller, doc_id=doc_id, text=body.text
    ).on_conflict_do_update(
        index_elements=[MaterialNote.owner, MaterialNote.doc_id],
        set_={"text": body.text},
    )
    await db.execute(stmt)
    await db.commit()
    return {"ok": True, "doc_id": doc_id, "text": body.text}
