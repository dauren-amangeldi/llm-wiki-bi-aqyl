"""Unit tests for Celery task definitions (tasks.py).

Tests use Celery's eager-execution mode (task_always_eager=True) so no broker
or worker is needed.  The pipeline coroutine itself is always mocked.

Key behaviours verified:
  - Permanent errors (NotFoundError, AuthenticationError) → propagate directly,
    no retry attempt.
  - Transient errors (RuntimeError)                       → Celery raises Retry
    (which in production schedules another attempt).
  - Success                                               → task returns normally.
"""

from unittest.mock import AsyncMock, patch

import httpx
import openai
import pytest

from llm_wiki.orchestrator.tasks import celery_app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_not_found_exc() -> openai.NotFoundError:
    req = httpx.Request("POST", "http://ollama:11434/v1/chat/completions")
    return openai.NotFoundError(
        "model 'qwen2.5-coder:14b' not found",
        response=httpx.Response(404, request=req),
        body={"error": {"message": "model not found"}},
    )


def _make_auth_exc() -> openai.AuthenticationError:
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return openai.AuthenticationError(
        "Incorrect API key",
        response=httpx.Response(401, request=req),
        body={"error": {"message": "invalid api key"}},
    )


@pytest.fixture(autouse=True)
def _eager_celery() -> None:  # type: ignore[misc]
    """Run tasks synchronously in-process; propagate exceptions immediately."""
    celery_app.conf.update(task_always_eager=True, task_eager_propagates=True)
    yield
    celery_app.conf.update(task_always_eager=False, task_eager_propagates=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_not_found_error_is_not_retried() -> None:
    """NotFoundError propagates immediately — Celery must NOT schedule a retry."""
    exc = _make_not_found_exc()

    # Patch process_file (the coroutine) to raise NotFoundError
    with patch(
        "llm_wiki.orchestrator.pipeline.process_file",
        new=AsyncMock(side_effect=exc),
    ):
        from llm_wiki.orchestrator.tasks import process_file_task

        with pytest.raises(openai.NotFoundError):
            process_file_task.delay("file-no-retry")

    # If we got here the task DID NOT turn the error into a Celery Retry,
    # which means no retry was attempted — the assertion is implicitly the
    # exception type check above.


def test_auth_error_is_not_retried() -> None:
    """AuthenticationError propagates immediately — no retry."""
    exc = _make_auth_exc()

    with patch(
        "llm_wiki.orchestrator.pipeline.process_file",
        new=AsyncMock(side_effect=exc),
    ):
        from llm_wiki.orchestrator.tasks import process_file_task

        with pytest.raises(openai.AuthenticationError):
            process_file_task.delay("file-bad-auth")


def test_transient_error_triggers_retry() -> None:
    """A RuntimeError raises celery.exceptions.Retry (scheduled for later)."""
    from celery.exceptions import Retry

    with patch(
        "llm_wiki.orchestrator.pipeline.process_file",
        new=AsyncMock(side_effect=RuntimeError("temporary glitch")),
    ):
        from llm_wiki.orchestrator.tasks import process_file_task

        # In eager+propagate mode self.retry() raises celery.exceptions.Retry
        with pytest.raises(Retry):
            process_file_task.delay("file-transient")


def test_successful_pipeline_returns_none() -> None:
    """A successful pipeline run returns None without raising."""
    with patch(
        "llm_wiki.orchestrator.pipeline.process_file",
        new=AsyncMock(return_value=None),
    ):
        from llm_wiki.orchestrator.tasks import process_file_task

        result = process_file_task.delay("file-ok")
        # In eager mode .get() returns the task result
        assert result.get() is None
