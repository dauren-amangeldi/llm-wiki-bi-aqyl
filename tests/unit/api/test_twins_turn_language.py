from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_wiki.agents.twins import ChatReplyResult, ChatRouteResult
from llm_wiki.api.v1 import twins
from llm_wiki.storage.metadata import CaseRecord, FileRecord, TwinPersona


@pytest.mark.parametrize("opening,expected", [(False, "en"), (True, "kk")])
async def test_language_is_kept_across_persona_handoffs(db_session, monkeypatch, opening, expected):
    db_session.add(CaseRecord(id="case", title="Кейс", owner="me", doc_ids=["ready"]))
    db_session.add(FileRecord(file_id="ready", original_name="ready.md", status="DONE", owner="me"))
    for pid in ("musk", "jobs"):
        db_session.add(TwinPersona(id=pid, name=pid, inspiration="x", real_name=pid,
                                  track="tech", lens="x", system_prompt="Русская персона", avatar_init="X"))
    await db_session.commit()
    agent = MagicMock()
    agent.route_message = AsyncMock(return_value=ChatRouteResult(["musk"], "en"))
    agent.respond_as_persona = AsyncMock(side_effect=[
        ChatReplyResult("musk", ["First reply"], "", "user", "jobs" if not opening else ""),
        ChatReplyResult("jobs", ["Second reply"], "", "musk", ""),
    ])
    client = MagicMock(aclose=AsyncMock())
    monkeypatch.setattr("llm_wiki.llm.client.LLMClient", lambda: client)
    monkeypatch.setattr(twins, "TwinsAgent", lambda _: agent)
    response = await twins.twin_chat_endpoint(
        twins.TwinChatRequest(case_id="case", persona_ids=["musk", "jobs"],
                              message="What are we missing?", language="kk", opening=opening),
        db_session, "me",
    )
    events = [chunk async for chunk in response.body_iterator]
    assert '"done": true' in events[-1]
    assert agent.respond_as_persona.await_count == 2
    assert [call.args[4] for call in agent.respond_as_persona.call_args_list] == [expected, expected]
    if opening:
        agent.route_message.assert_not_called()
    else:
        assert agent.route_message.call_args.kwargs["latest_question"] == "What are we missing?"
    client.aclose.assert_awaited_once()
