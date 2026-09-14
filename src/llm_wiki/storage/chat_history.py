"""Serialize clearing and saving a chat, without holding locks during an LLM call."""

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.storage.metadata import ChatRecord, ChatScopeRecord


async def lock_chat_scope(
    session: AsyncSession,
    *,
    user_key: str,
    scope_type: str,
    scope_id: str,
) -> ChatScopeRecord:
    key = {"user_key": user_key, "scope_type": scope_type, "scope_id": scope_id}
    await session.execute(
        insert(ChatScopeRecord).values(**key, revision=0).on_conflict_do_nothing()
    )
    return (
        await session.scalars(
            select(ChatScopeRecord)
            .filter_by(**key)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
    ).one()


async def begin_chat_turn(
    session: AsyncSession,
    *,
    user_key: str,
    scope_type: str,
    scope_id: str,
) -> int:
    scope = await lock_chat_scope(
        session, user_key=user_key, scope_type=scope_type, scope_id=scope_id
    )
    revision = scope.revision
    await session.commit()
    return revision


async def clear_history(
    session: AsyncSession,
    *,
    user_key: str,
    scope_type: str,
    scope_id: str,
) -> int:
    scope = await lock_chat_scope(
        session, user_key=user_key, scope_type=scope_type, scope_id=scope_id
    )
    scope.revision += 1
    result = await session.execute(
        delete(ChatRecord).where(
            ChatRecord.user_key == user_key,
            ChatRecord.scope_type == scope_type,
            ChatRecord.scope_id == scope_id,
        )
    )
    await session.commit()
    return result.rowcount
