import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_wiki.agents.twin_summary import generate_positions


@pytest.mark.parametrize("language,word", [("ru", "Russian"), ("en", "English"), ("kk", "Kazakh")])
async def test_one_structured_call_in_requested_language(language, word):
    llm = MagicMock(
        complete=AsyncMock(
            return_value=(
                json.dumps(
                    {
                        "jobs": {"position": "A pilot first.", "key_argument": "Learn from users."},
                        "musk": {"position": "Test costs.", "key_argument": "Control spending."},
                    }
                ),
                None,
            )
        )
    )
    transcript = [
        {"speaker": "jobs", "role": "expert", "text": "Run a pilot"},
        {"speaker": "musk", "role": "expert", "text": "Test costs"},
    ]
    result = await generate_positions(llm, transcript, language, "session")
    assert len(result) == 2
    llm.complete.assert_awaited_once()
    kwargs = llm.complete.call_args.kwargs
    assert word in kwargs["system"]
    assert kwargs["json_schema"]["required"] == ["jobs", "musk"]
    assert kwargs["file_id"] == "session"
    assert "Run a pilot" in kwargs["prompt"]


@pytest.mark.parametrize(
    "response",
    [
        "{}",
        "[]",
        '{"jobs":""}',
        '{"jobs":"Не удалось получить ответ."}',
        '{"jobs":"ok","other":"invented"}',
        '{"jobs":12}',
        '{"jobs":{"position":"A pilot first."}}',
        '{"jobs":{"position":"A pilot first.","key_argument":" "}}',
        '{"jobs":{"position":"A pilot first.","key_argument":12}}',
        '{"jobs":{"position":"A pilot first.","key_argument":"Could not get a response."}}',
        '{"jobs":{"position":"A pilot first.","key_argument":"Learn.","extra":"Invented"}}',
        "not json",
    ],
)
async def test_invalid_positions_are_not_accepted(response):
    llm = MagicMock(complete=AsyncMock(return_value=(response, None)))
    with pytest.raises(ValueError):
        await generate_positions(
            llm, [{"speaker": "jobs", "role": "expert", "text": "Test."}], "en", "sid"
        )
