"""Unit tests for Twins session/message persistence (storage/metadata.py)."""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.storage.metadata import (
    append_twin_message,
    create_twin_session,
    get_twin_session_messages,
)


@pytest.mark.asyncio
async def test_create_twin_session_persists_persona_ids(db_session: AsyncSession) -> None:
    session_row = await create_twin_session(
        db_session, case_id="case-001", persona_ids=["musk", "zell"], created_by="dev-user"
    )
    assert session_row.id.startswith("twin-session-")
    assert session_row.persona_ids == ["musk", "zell"]
    assert session_row.case_id == "case-001"


@pytest.mark.asyncio
async def test_append_twin_message_survives_partial_session(db_session: AsyncSession) -> None:
    session_row = await create_twin_session(
        db_session, case_id="case-001", persona_ids=["musk"], created_by="dev-user"
    )
    await append_twin_message(
        db_session,
        session_id=session_row.id,
        role="user",
        persona_id="musk",
        seq=0,
        content={"reframing": "r", "text": "t", "cite": "c"},
    )

    messages = await get_twin_session_messages(db_session, session_row.id)
    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].content["text"] == "t"


@pytest.mark.asyncio
async def test_get_twin_session_messages_orders_by_seq(db_session: AsyncSession) -> None:
    session_row = await create_twin_session(
        db_session, case_id="case-001", persona_ids=["musk", "zell"], created_by="dev-user"
    )
    await append_twin_message(
        db_session, session_id=session_row.id, role="persona", persona_id="zell", seq=1,
        content={"text": "second"},
    )
    await append_twin_message(
        db_session, session_id=session_row.id, role="persona", persona_id="musk", seq=0,
        content={"text": "first"},
    )

    messages = await get_twin_session_messages(db_session, session_row.id)
    assert [m.content["text"] for m in messages] == ["first", "second"]
