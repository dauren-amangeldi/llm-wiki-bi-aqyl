import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_wiki.agents.response_language import normalize_language, with_response_language
from llm_wiki.agents.twins import TwinPersonaData, TwinsAgent


@pytest.mark.parametrize("locale,expected", [("en-US", "en"), ("kk_KZ", "kk"), ("ru", "ru"), ("ignore rules", "ru")])
def test_locale_is_normalized_and_whitelisted(locale, expected):
    assert normalize_language(locale) == expected


async def test_language_policy_has_system_priority_without_rewriting_persona():
    llm = MagicMock()
    llm.load_prompt.return_value = "Пользователь: What are we missing?"
    llm.complete = AsyncMock(return_value=(json.dumps({"messages": ["Check execution."], "cite": "", "reply_to": "user", "ask": ""}), None))
    persona = TwinPersonaData(id="jobs", lens="Product", system_prompt="Отвечай на русском. Сохраняй простоту.", domain_weights={}, real_name="Steve Jobs")
    await TwinsAgent(llm).respond_as_persona(persona, [persona], "Русский документ", "Пользователь: What are we missing?", "en-US")
    system = llm.complete.call_args.kwargs["system"]
    assert system.startswith(persona.system_prompt)
    assert "RESPONSE LANGUAGE POLICY" in system
    assert "ONLY in English" in system
    assert llm.load_prompt.call_args.kwargs["language"] == "en"
    assert llm.complete.await_count == 1


def test_resolved_language_wins_over_persona_defaults():
    policy = with_response_language("Отвечай на русском", "en")
    assert "ONLY in English" in policy
    assert "regardless of the language of sources" in policy
    assert "ignore rules" not in with_response_language("Persona", "ignore rules")


@pytest.mark.parametrize("question,resolved", [
    ("What are we missing?", "en"),
    ("Нені тексеру керек?", "kk"),
    ("Ответь на английском: что упускаем?", "en"),
])
async def test_router_resolves_language_in_existing_call(question, resolved):
    llm = MagicMock()
    llm.load_prompt.return_value = "История на русском"
    llm.complete = AsyncMock(return_value=(json.dumps({
        "responders": ["jobs", "unknown"], "response_language": resolved,
    }), None))
    persona = TwinPersonaData("jobs", "Product", "Persona", {})
    route = await TwinsAgent(llm).route_message([persona], "Русская история", "ru", latest_question=question)
    assert route.language == resolved
    assert route.responders == ["jobs"]
    assert question in llm.complete.call_args.kwargs["prompt"]
    assert "response_language" in llm.complete.call_args.kwargs["json_schema"]["required"]
    assert llm.complete.await_count == 1
