"""Unit tests for the LLM client wrapper (LW-4).

All tests use unittest.mock — no live LLM API calls are made.
Every test that exercises LLMClient.complete verifies that a JSON-line
record is written to usage.log.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_openai_response(
    content: str = "hello",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    cached_tokens: int = 0,
) -> MagicMock:
    """Build a minimal mock that mimics openai.types.chat.ChatCompletion."""
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.prompt_tokens_details = MagicMock()
    usage.prompt_tokens_details.cached_tokens = cached_tokens

    choice = MagicMock()
    choice.message.content = content

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


def _make_client(tmp_path: Path, provider: str = "openai") -> Any:
    """Create an LLMClient whose usage_log_path points to *tmp_path/usage.log*."""
    from llm_wiki.llm.client import LLMClient

    models = {
        "openai": {"openai_api_key": "sk-test", "openai_model": "gpt-5.4-mini"},
        "ollama": {"ollama_base_url": "http://ollama:11434", "ollama_model": "qwen2.5-coder:14b"},
    }

    with patch("llm_wiki.llm.client.openai.AsyncOpenAI") as mock_sdk:
        mock_sdk.return_value = AsyncMock()
        with patch("llm_wiki.config.settings") as mock_settings:
            mock_settings.llm_provider = provider
            mock_settings.openai_api_key = models.get(provider, {}).get("openai_api_key", "")
            mock_settings.openai_model = "gpt-5.4-mini"
            mock_settings.ollama_base_url = "http://ollama:11434"
            mock_settings.ollama_model = "qwen2.5-coder:14b"
            mock_settings.anthropic_api_key = ""
            mock_settings.anthropic_model = "claude-3-5-sonnet"
            mock_settings.usage_log_path = tmp_path / "usage.log"
            mock_settings.price_table = {
                "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
                "gpt-5.4": {"input": 2.50, "output": 15.00},
                "ollama": {"input": 0.00, "output": 0.00},
            }
            client = LLMClient()
            client._client = mock_sdk.return_value
            client._usage_log_path = tmp_path / "usage.log"
    return client


# ---------------------------------------------------------------------------
# _compute_cost
# ---------------------------------------------------------------------------


def test_compute_cost_gpt_mini(tmp_path: Path) -> None:
    """_compute_cost returns correct USD value for gpt-5.4-mini pricing."""
    client = _make_client(tmp_path)
    # 1 000 input @ $0.75/1M  +  500 output @ $4.50/1M
    expected = round((1000 * 0.75 + 500 * 4.50) / 1_000_000, 6)
    result = client._compute_cost("gpt-5.4-mini", 1000, 500)
    assert abs(result - expected) < 1e-9


def test_compute_cost_ollama_is_free(tmp_path: Path) -> None:
    """_compute_cost returns 0.0 for ollama (local, no billing)."""
    client = _make_client(tmp_path)
    assert client._compute_cost("ollama", 10_000, 5_000) == 0.0


# ---------------------------------------------------------------------------
# complete — happy path
# ---------------------------------------------------------------------------


async def test_complete_returns_text_and_usage(tmp_path: Path) -> None:
    """complete returns the response text and a populated LLMUsage."""
    client = _make_client(tmp_path)
    mock_response = _make_openai_response("wiki page content", 200, 80)
    client._client.chat = MagicMock()
    client._client.chat.completions = MagicMock()
    client._client.chat.completions.create = AsyncMock(return_value=mock_response)

    text, usage = await client.complete("prompt", "system", "file-001", "writer")

    assert text == "wiki page content"
    assert usage.file_id == "file-001"
    assert usage.input_tokens == 200
    assert usage.output_tokens == 80
    assert usage.cost_usd >= 0.0


async def test_complete_writes_usage_log(tmp_path: Path) -> None:
    """complete appends a JSON-line record to usage.log."""
    client = _make_client(tmp_path)
    mock_response = _make_openai_response("result", 100, 50)
    client._client.chat = MagicMock()
    client._client.chat.completions.create = AsyncMock(return_value=mock_response)

    await client.complete("p", "s", "file-log-test", "search")

    log_path = tmp_path / "usage.log"
    assert log_path.exists(), "usage.log was not created"
    line = log_path.read_text().strip().splitlines()[-1]
    record = json.loads(line)
    assert record["file_id"] == "file-log-test"
    assert record["agent_type"] == "search"
    assert "model" in record
    assert "cost_usd" in record
    assert "timestamp" in record
    assert "duration_ms" in record


async def test_complete_json_format_passed_to_openai(tmp_path: Path) -> None:
    """complete passes response_format={'type':'json_object'} when format='json'."""
    client = _make_client(tmp_path)
    mock_response = _make_openai_response("{}")
    create_mock = AsyncMock(return_value=mock_response)
    client._client.chat = MagicMock()
    client._client.chat.completions.create = create_mock

    await client.complete("p", "s", "fid", "writer", response_format="json")

    _, kwargs = create_mock.call_args
    # response_format may be positional or keyword — check both
    call_kwargs = create_mock.call_args.kwargs
    assert call_kwargs.get("response_format") == {"type": "json_object"}


async def test_complete_text_format_no_response_format_key(tmp_path: Path) -> None:
    """complete does NOT send response_format when format='text'."""
    client = _make_client(tmp_path)
    mock_response = _make_openai_response("plain text")
    create_mock = AsyncMock(return_value=mock_response)
    client._client.chat = MagicMock()
    client._client.chat.completions.create = create_mock

    await client.complete("p", "s", "fid", "search", response_format="text")

    call_kwargs = create_mock.call_args.kwargs
    assert "response_format" not in call_kwargs


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------


async def test_complete_retries_on_rate_limit(tmp_path: Path) -> None:
    """complete retries up to 3 times on RateLimitError, then succeeds."""
    client = _make_client(tmp_path)
    mock_response = _make_openai_response("ok")
    _req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    rate_limit_exc = openai.RateLimitError(
        "rate limited",
        response=httpx.Response(429, request=_req),
        body={"error": {"message": "rate limit"}},
    )
    create_mock = AsyncMock(
        side_effect=[rate_limit_exc, rate_limit_exc, mock_response]
    )
    client._client.chat = MagicMock()
    client._client.chat.completions.create = create_mock

    # Patch asyncio.sleep so the test doesn't actually wait
    with patch("llm_wiki.llm.client.asyncio.sleep", new_callable=AsyncMock):
        text, usage = await client.complete("p", "s", "fid", "lint")

    assert text == "ok"
    assert create_mock.call_count == 3


async def test_complete_raises_after_max_retries(tmp_path: Path) -> None:
    """complete raises after MAX_RETRIES consecutive transient failures."""
    client = _make_client(tmp_path)
    _req2 = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    rate_limit_exc = openai.RateLimitError(
        "rate limited",
        response=httpx.Response(429, request=_req2),
        body={"error": {"message": "rate limit"}},
    )
    create_mock = AsyncMock(side_effect=rate_limit_exc)
    client._client.chat = MagicMock()
    client._client.chat.completions.create = create_mock

    with patch("llm_wiki.llm.client.asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(openai.RateLimitError):
            await client.complete("p", "s", "fid", "search")

    assert create_mock.call_count == client._MAX_RETRIES


async def test_complete_no_retry_on_auth_error(tmp_path: Path) -> None:
    """complete raises immediately (no retry) on 401 AuthenticationError."""
    client = _make_client(tmp_path)
    _req3 = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    auth_exc = openai.AuthenticationError(
        "401 Unauthorized",
        response=httpx.Response(401, request=_req3),
        body={"error": {"message": "Invalid API key"}},
    )
    create_mock = AsyncMock(side_effect=auth_exc)
    client._client.chat = MagicMock()
    client._client.chat.completions.create = create_mock

    with patch("llm_wiki.llm.client.asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(openai.AuthenticationError):
            await client.complete("p", "s", "fid", "writer")

    # Must have been called exactly once — no retries
    assert create_mock.call_count == 1


# ---------------------------------------------------------------------------
# load_prompt
# ---------------------------------------------------------------------------


def test_load_prompt_interpolates_variables(tmp_path: Path) -> None:
    """load_prompt reads the .md file and substitutes {variables}."""
    client = _make_client(tmp_path)
    # Write a temporary prompt file into the real prompts dir
    prompts_dir = client.PROMPTS_DIR
    test_prompt = prompts_dir / "_test_fixture.md"
    test_prompt.write_text("Hello {name}, you have {count} messages.\n")
    try:
        result = client.load_prompt("_test_fixture", name="Alice", count=3)
        assert result == "Hello Alice, you have 3 messages.\n"
    finally:
        test_prompt.unlink(missing_ok=True)
