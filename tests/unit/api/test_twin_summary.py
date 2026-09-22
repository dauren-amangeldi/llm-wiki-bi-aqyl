import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.api.v1 import twins
from llm_wiki.storage.metadata import (
    TwinMessage,
    TwinSummary,
    append_twin_message,
    create_twin_session,
    delete_twin_session,
)


@pytest.fixture
async def council(db_session, monkeypatch):
    row = await create_twin_session(
        db_session, case_id="case", persona_ids=["jobs"], created_by="owner"
    )
    await append_twin_message(
        db_session,
        session_id=row.id,
        role="persona",
        persona_id="jobs",
        seq=0,
        content={"text": "Start with a small pilot and test customer demand."},
    )
    llm = MagicMock(aclose=AsyncMock())
    monkeypatch.setattr("llm_wiki.llm.client.LLMClient", lambda: llm)
    generate = AsyncMock(return_value=[{"persona_id": "jobs", "position": "Run a pilot first."}])
    monkeypatch.setattr(twins, "generate_positions", generate)
    return row.id, generate, llm


async def test_reading_never_generates_and_result_survives_reopening(db_session, council):
    sid, generate, llm = council
    assert await twins.get_twin_summary(sid, db_session, "owner") == {"summary": None}
    generate.assert_not_called()
    first = await twins.summarize_twin_session(
        sid, twins.TwinSummaryRequest(language="en"), db_session, "owner"
    )
    assert first["summary"]["positions"] == [
        {"persona_id": "jobs", "position": "Run a pilot first."}
    ]
    assert first["summary"]["source_seq"] == 0
    assert first["summary"]["language"] == "en"
    assert first == await twins.get_twin_summary(sid, db_session, "owner")
    assert first == await twins.summarize_twin_session(
        sid, twins.TwinSummaryRequest(), db_session, "owner"
    )
    generate.assert_awaited_once()
    llm.aclose.assert_awaited_once()
    # The summary is not a chat turn and does not grow the next model context.
    assert len((await db_session.scalars(select(TwinMessage))).all()) == 1


async def test_new_message_keeps_snapshot_until_explicit_regeneration(db_session, council):
    sid, generate, _ = council
    first = await twins.summarize_twin_session(sid, twins.TwinSummaryRequest(), db_session, "owner")
    await append_twin_message(
        db_session,
        session_id=sid,
        role="persona",
        persona_id="jobs",
        seq=1,
        content={"text": "Check access controls next."},
    )
    assert first == await twins.get_twin_summary(sid, db_session, "owner")
    regenerate = twins.TwinSummaryRequest(
        previous_revision=first["summary"]["revision"], language="kk"
    )
    second = await twins.summarize_twin_session(sid, regenerate, db_session, "owner")
    assert second["summary"]["revision"] != first["summary"]["revision"]
    assert second["summary"]["source_seq"] == 1
    # Retrying the same request after it finished must not spend again.
    assert second == await twins.summarize_twin_session(sid, regenerate, db_session, "owner")
    assert generate.await_count == 2


@pytest.mark.parametrize(
    "text,failed",
    [
        ("Не удалось получить ответ.", False),
        ("Could not get a response.", False),
        ("Жауап алу мүмкін болмады.", False),
        ("…", False),
        (" ", False),
        ("Provider unavailable", True),
    ],
)
async def test_failed_replies_cannot_be_summarized(db_session, council, text, failed):
    sid, generate, _ = council
    msg = (await db_session.scalars(select(TwinMessage))).one()
    msg.content = {"text": text, "failed": failed}
    await db_session.commit()
    with pytest.raises(HTTPException) as exc:
        await twins.summarize_twin_session(sid, twins.TwinSummaryRequest(), db_session, "owner")
    assert exc.value.status_code == 422
    generate.assert_not_called()
    assert await db_session.get(TwinSummary, sid) is None


async def test_partial_failure_is_excluded(db_session, council):
    sid, generate, _ = council
    await append_twin_message(
        db_session,
        session_id=sid,
        role="persona",
        persona_id="musk",
        seq=1,
        content={"text": "Не удалось получить ответ."},
    )
    await twins.summarize_twin_session(sid, twins.TwinSummaryRequest(), db_session, "owner")
    assert [m["speaker"] for m in generate.call_args.args[1]] == ["jobs"]


async def test_errors_keep_saved_summary_and_hide_provider_details(db_session, council):
    sid, generate, llm = council
    first = await twins.summarize_twin_session(sid, twins.TwinSummaryRequest(), db_session, "owner")
    await append_twin_message(
        db_session,
        session_id=sid,
        role="persona",
        persona_id="jobs",
        seq=1,
        content={"text": "A new substantive reply."},
    )
    generate.side_effect = RuntimeError("sensitive provider response")
    with pytest.raises(HTTPException) as exc:
        await twins.summarize_twin_session(
            sid,
            twins.TwinSummaryRequest(previous_revision=first["summary"]["revision"]),
            db_session,
            "owner",
        )
    assert (exc.value.status_code, exc.value.detail) == (503, "summary_failed")
    assert first == await twins.get_twin_summary(sid, db_session, "owner")
    assert llm.aclose.await_count == 2


async def test_private_summary_cannot_be_read_or_generated_by_other_user(db_session, council):
    sid, generate, _ = council
    for call in [
        twins.get_twin_summary(sid, db_session, "other"),
        twins.summarize_twin_session(sid, twins.TwinSummaryRequest(), db_session, "other"),
    ]:
        with pytest.raises(HTTPException) as exc:
            await call
        assert exc.value.status_code == 404
    generate.assert_not_called()


async def test_concurrent_clicks_have_one_model_call(db_session, db_engine, council):
    sid, generate, _ = council
    entered, finish = asyncio.Event(), asyncio.Event()

    async def slow(*args):
        entered.set()
        await finish.wait()
        return [{"persona_id": "jobs", "position": "Pilot."}]

    generate.side_effect = slow
    first = asyncio.create_task(
        twins.summarize_twin_session(sid, twins.TwinSummaryRequest(), db_session, "owner")
    )
    await asyncio.wait_for(entered.wait(), 5)
    try:
        async with AsyncSession(db_engine) as other:
            with pytest.raises(HTTPException) as exc:
                await twins.summarize_twin_session(sid, twins.TwinSummaryRequest(), other, "owner")
            assert exc.value.status_code == 409
    finally:
        finish.set()
        await first
    generate.assert_awaited_once()


async def test_snapshot_does_not_claim_new_messages_generated_during_request(
    db_session, db_engine, council
):
    sid, generate, _ = council

    async def add_message(*args):
        async with AsyncSession(db_engine, expire_on_commit=False) as other:
            await append_twin_message(
                other,
                session_id=sid,
                role="user",
                persona_id=None,
                seq=1,
                content={"text": "New question"},
            )
        return [{"persona_id": "jobs", "position": "Pilot."}]

    generate.side_effect = add_message
    result = await twins.summarize_twin_session(
        sid, twins.TwinSummaryRequest(), db_session, "owner"
    )
    assert result["summary"]["source_seq"] == 0


async def test_deleting_session_removes_summary(db_session, council):
    sid, _, _ = council
    await twins.summarize_twin_session(sid, twins.TwinSummaryRequest(), db_session, "owner")
    await delete_twin_session(db_session, sid)
    assert (await db_session.scalars(select(TwinSummary))).all() == []


async def test_deleting_during_generation_does_not_resurrect_session(
    db_session, db_engine, council
):
    sid, generate, _ = council

    async def delete(*args):
        async with AsyncSession(db_engine) as other:
            await delete_twin_session(other, sid)
        return [{"persona_id": "jobs", "position": "Pilot."}]

    generate.side_effect = delete
    with pytest.raises(HTTPException) as exc:
        await twins.summarize_twin_session(sid, twins.TwinSummaryRequest(), db_session, "owner")
    assert exc.value.status_code == 404
    assert (await db_session.scalars(select(TwinSummary))).all() == []


async def test_long_transcript_fails_before_model_call(db_session, council):
    sid, generate, _ = council
    await append_twin_message(
        db_session,
        session_id=sid,
        role="user",
        persona_id=None,
        seq=1,
        content={"text": "x" * 80001},
    )
    with pytest.raises(HTTPException) as exc:
        await twins.summarize_twin_session(sid, twins.TwinSummaryRequest(), db_session, "owner")
    assert exc.value.detail == "summary_too_long"
    generate.assert_not_called()


@pytest.mark.parametrize(
    "role,content",
    [
        ("user", {"text": "Another question"}),
        ("persona", {"text": "Не удалось получить ответ."}),
        ("persona", {"text": "Provider failed", "failed": True}),
    ],
)
async def test_cooldown_rejects_user_questions_and_failed_expert_replies(
    db_session, council, role, content
):
    sid, generate, _ = council
    first = await twins.summarize_twin_session(sid, twins.TwinSummaryRequest(), db_session, "owner")
    await append_twin_message(
        db_session,
        session_id=sid,
        role=role,
        persona_id="jobs" if role == "persona" else None,
        seq=1,
        content=content,
    )
    with pytest.raises(HTTPException) as exc:
        await twins.summarize_twin_session(
            sid,
            twins.TwinSummaryRequest(previous_revision=first["summary"]["revision"]),
            db_session,
            "owner",
        )
    assert (exc.value.status_code, exc.value.detail) == (429, "summary_cooldown")
    assert 0 < int(exc.value.headers["Retry-After"]) <= 900
    generate.assert_awaited_once()
    assert first == await twins.get_twin_summary(sid, db_session, "owner")


async def test_cooldown_expires_after_fifteen_minutes(db_session, council):
    sid, generate, _ = council
    first = await twins.summarize_twin_session(sid, twins.TwinSummaryRequest(), db_session, "owner")
    saved = await db_session.get(TwinSummary, sid)
    saved.created_at = datetime.now(timezone.utc) - timedelta(minutes=15, seconds=1)
    await db_session.commit()
    result = await twins.summarize_twin_session(
        sid,
        twins.TwinSummaryRequest(previous_revision=first["summary"]["revision"]),
        db_session,
        "owner",
    )
    assert result["summary"]["revision"] != first["summary"]["revision"]
    assert generate.await_count == 2


async def test_valid_new_expert_reply_unlocks_early_refresh(db_session, council):
    sid, generate, _ = council
    first = await twins.summarize_twin_session(sid, twins.TwinSummaryRequest(), db_session, "owner")
    await append_twin_message(
        db_session,
        session_id=sid,
        role="persona",
        persona_id="jobs",
        seq=1,
        content={"text": "Test with five customers first."},
    )
    result = await twins.summarize_twin_session(
        sid,
        twins.TwinSummaryRequest(previous_revision=first["summary"]["revision"]),
        db_session,
        "owner",
    )
    assert result["summary"]["source_seq"] == 1
    assert result["summary"]["revision"] != first["summary"]["revision"]
    assert generate.await_count == 2
