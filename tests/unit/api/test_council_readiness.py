"""Readiness is enforced before any council inference, including old sessions."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.agents.twins import ChatReplyResult
from llm_wiki.api.council_readiness import require_ready_case
from llm_wiki.api.v1 import twins
from llm_wiki.storage.metadata import (
    CaseRecord,
    FileRecord,
    TwinMessage,
    TwinPersona,
    TwinSession,
    TwinSummary,
    append_twin_message,
    create_twin_session,
)


@pytest.fixture
async def context(db_session, monkeypatch):
    case = CaseRecord(id="case", title="Case", owner="me", doc_ids=[])
    db_session.add(case)
    for pid in ("jobs", "musk"):
        db_session.add(
            TwinPersona(
                id=pid,
                name=pid,
                real_name=pid,
                inspiration="x",
                track="tech",
                lens="x",
                system_prompt="x",
                avatar_init="X",
            )
        )
    await db_session.commit()
    client = MagicMock(aclose=AsyncMock())
    constructor = MagicMock(return_value=client)
    monkeypatch.setattr("llm_wiki.llm.client.LLMClient", constructor)
    return case, constructor


async def add_file(db, case, status, fid="file"):
    row = FileRecord(file_id=fid, original_name=f"{fid}.md", status=status, owner="me")
    db.add(row)
    case.doc_ids = [*case.doc_ids, fid]
    await db.commit()
    return row


@pytest.mark.parametrize(
    "status", [None, "RECEIVED", "STORED", "SEARCHED", "WRITTEN", "LINTED", "LOGGED", "FAILED"]
)
async def test_zero_ready_blocks_opening_messages_summaries_and_lineup(db_session, context, status):
    case, constructor = context
    if status:
        await add_file(db_session, case, status)
    session = await create_twin_session(
        db_session, case_id=case.id, persona_ids=["jobs"], created_by="me"
    )
    sid = session.id
    await append_twin_message(
        db_session,
        session_id=sid,
        role="persona",
        persona_id="jobs",
        seq=0,
        content={"text": "Previous answer"},
    )
    db_session.add(
        TwinSummary(
            session_id=sid,
            created_at=datetime.now(timezone.utc),
            revision="old",
            source_seq=0,
            language="en",
            positions=[{"persona_id": "jobs", "position": "Keep me"}],
        )
    )
    await db_session.commit()
    old_summary = await twins.get_twin_summary(sid, db_session, "me")

    for opening, session_id in [(True, None), (False, sid)]:
        with pytest.raises(HTTPException) as error:
            await twins.twin_chat_endpoint(
                twins.TwinChatRequest(
                    case_id="case",
                    persona_ids=["jobs"],
                    opening=opening,
                    session_id=session_id,
                    message="Discuss this",
                ),
                db_session,
                "me",
            )
        assert (error.value.status_code, error.value.detail) == (422, "council_no_ready_materials")
    with pytest.raises(HTTPException, match="council_no_ready_materials"):
        await twins.summarize_twin_session(
            sid, twins.TwinSummaryRequest(previous_revision="old"), db_session, "me"
        )
    with pytest.raises(HTTPException, match="council_no_ready_materials"):
        await twins.patch_twin_session_personas(
            sid, twins.SessionPersonasRequest(persona_ids=["musk"]), db_session, "me"
        )

    constructor.assert_not_called()
    assert await db_session.scalar(select(func.count()).select_from(TwinSession)) == 1
    transcript = await twins.get_twin_session_transcript(sid, db_session, "me")
    assert len(transcript) == 1
    assert transcript[0]["content"]["text"] == "Previous answer"
    assert await twins.get_twin_summary(sid, db_session, "me") == old_summary


async def test_readiness_tracks_completion_removal_failures_and_visibility(db_session, context):
    case, _ = context
    pending = await add_file(db_session, case, "STORED")
    await add_file(db_session, case, "FAILED", "failed")
    other = await add_file(db_session, case, "DONE", "private")
    other.sensitive, other.owner = True, "someone-else"
    case.doc_ids = [*case.doc_ids, "missing"]
    db_session.add(
        CaseRecord(id="hidden", title="Hidden", owner="someone-else", sensitive=True, doc_ids=[])
    )
    await db_session.commit()
    snapshot = await twins.get_council_readiness(db_session, "me")
    assert "hidden" not in snapshot
    assert snapshot[case.id] == {
        "ready_doc_ids": [],
        "processing_doc_ids": ["file"],
        "doc_ids": case.doc_ids,
        "sensitive": True,
    }
    pending.status = "DONE"
    await db_session.commit()
    assert (await twins.get_council_readiness(db_session, "me"))[case.id]["ready_doc_ids"] == [
        "file"
    ]
    _, ready = await require_ready_case(db_session, case.id, "me")
    assert [f.file_id for f in ready] == ["file"]
    # A removed membership must not be restored by a stale ORM identity map.
    async with AsyncSession(db_session.bind) as another:
        await another.execute(
            update(CaseRecord).where(CaseRecord.id == "case").values(doc_ids=["failed", "private"])
        )
        await another.commit()
    with pytest.raises(HTTPException, match="council_no_ready_materials"):
        await require_ready_case(db_session, case.id, "me")
    with pytest.raises(HTTPException, match="case_not_available"):
        await require_ready_case(db_session, "hidden", "me")


async def test_only_ready_materials_reach_experts(db_session, context, monkeypatch):
    case, constructor = context
    await add_file(db_session, case, "DONE", "ready")
    await add_file(db_session, case, "LOGGED", "pending")
    await add_file(db_session, case, "FAILED", "failed")
    load = MagicMock(return_value="Ready source context")
    monkeypatch.setattr(twins, "load_case_context", load)
    agent = MagicMock(
        respond_as_persona=AsyncMock(return_value=ChatReplyResult("jobs", ["Answer"], "", "", ""))
    )
    monkeypatch.setattr(twins, "TwinsAgent", lambda _: agent)
    response = await twins.twin_chat_endpoint(
        twins.TwinChatRequest(case_id=case.id, persona_ids=["jobs"], opening=True), db_session, "me"
    )
    events = [event async for event in response.body_iterator]
    assert '"done": true' in events[-1]
    assert [f.file_id for f in load.call_args.args[0]] == ["ready"]
    assert agent.respond_as_persona.call_args.args[2] == "Ready source context"
    constructor.assert_called_once()


async def test_removal_during_reply_stops_next_expert_and_discards_inflight_answer(
    db_session, context, monkeypatch
):
    case, _ = context
    await add_file(db_session, case, "DONE")

    async def reply(*args):
        async with AsyncSession(db_session.bind) as another:
            await another.execute(
                update(CaseRecord).where(CaseRecord.id == "case").values(doc_ids=[])
            )
            await another.commit()
        return ChatReplyResult("jobs", ["Answer"], "", "", "musk")

    agent = MagicMock(respond_as_persona=AsyncMock(side_effect=reply))
    monkeypatch.setattr(twins, "TwinsAgent", lambda _: agent)
    monkeypatch.setattr(twins, "load_case_context", lambda *a, **kw: "Context")
    response = await twins.twin_chat_endpoint(
        twins.TwinChatRequest(case_id=case.id, persona_ids=["jobs", "musk"], opening=True),
        db_session,
        "me",
    )
    events = [event async for event in response.body_iterator]
    assert '"error": "council_no_ready_materials"' in events[-1]
    agent.respond_as_persona.assert_awaited_once()
    assert await db_session.scalar(select(func.count()).select_from(TwinMessage)) == 0


async def test_removal_during_summary_preserves_saved_positions(db_session, context, monkeypatch):
    case, _ = context
    await add_file(db_session, case, "DONE")
    session = await create_twin_session(
        db_session, case_id=case.id, persona_ids=["jobs"], created_by="me"
    )
    sid = session.id
    await append_twin_message(
        db_session,
        session_id=sid,
        role="persona",
        persona_id="jobs",
        seq=1,
        content={"text": "New answer"},
    )
    db_session.add(
        TwinSummary(
            session_id=sid,
            created_at=datetime.now(timezone.utc),
            revision="old",
            source_seq=0,
            language="en",
            positions=[{"persona_id": "jobs", "position": "Old position"}],
        )
    )
    await db_session.commit()
    before = await twins.get_twin_summary(sid, db_session, "me")

    async def generate(*args):
        async with AsyncSession(db_session.bind) as another:
            await another.execute(
                update(CaseRecord).where(CaseRecord.id == "case").values(doc_ids=[])
            )
            await another.commit()
        return [{"persona_id": "jobs", "position": "Discard this"}]

    monkeypatch.setattr(twins, "generate_positions", generate)
    with pytest.raises(HTTPException, match="council_no_ready_materials"):
        await twins.summarize_twin_session(
            sid, twins.TwinSummaryRequest(previous_revision="old"), db_session, "me"
        )
    assert await twins.get_twin_summary(sid, db_session, "me") == before
