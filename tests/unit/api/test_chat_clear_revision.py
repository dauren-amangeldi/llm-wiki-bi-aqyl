from sqlalchemy import select
from llm_wiki.api.v1.ask import DocAskResponse, _persist_turn
from llm_wiki.storage.chat_history import begin_chat_turn, clear_history
from llm_wiki.storage.metadata import ChatRecord


async def test_clear_does_not_allow_an_inflight_answer_to_restore_history(db_session):
    key = {"user_key": "alice", "scope_type": "case", "scope_id": "case-1"}
    revision = await begin_chat_turn(db_session, **key)
    response = DocAskResponse(answer="Old answer")
    await _persist_turn(db_session, **key, question="Old question", response=response, revision=revision)
    other = {**key, "user_key": "bob"}
    other_revision = await begin_chat_turn(db_session, **other)
    await _persist_turn(db_session, **other, question="Bob's question", response=response, revision=other_revision)
    assert await clear_history(db_session, **key) == 2
    await _persist_turn(db_session, **key, question="Late question", response=response, revision=revision)
    rows = (await db_session.scalars(select(ChatRecord))).all()
    assert [r.user_key for r in rows] == ["bob", "bob"]
    new_revision = await begin_chat_turn(db_session, **key)
    assert new_revision > revision
    await _persist_turn(db_session, **key, question="New question", response=DocAskResponse(answer="New answer"), revision=new_revision)
    rows = (await db_session.scalars(select(ChatRecord).where(ChatRecord.user_key == "alice"))).all()
    assert {r.text for r in rows} == {"New question", "New answer"}


async def test_clear_is_scoped_and_idempotent(db_session):
    key = {"user_key": "alice", "scope_type": "case", "scope_id": "one"}
    for scope in ["one", "two"]:
        db_session.add(ChatRecord(**{**key, "scope_id": scope}, role="user", text=scope))
    db_session.add(ChatRecord(**{**key, "scope_type": "document"}, role="user", text="document"))
    await db_session.commit()
    assert await clear_history(db_session, **key) == 1
    assert await clear_history(db_session, **key) == 0
    assert {r.text for r in (await db_session.scalars(select(ChatRecord))).all()} == {"two", "document"}
