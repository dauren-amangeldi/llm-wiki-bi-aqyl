"""Tests for the llm_call_log table and its write helper."""

from collections.abc import Iterator
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.ext.asyncio import AsyncSession

import llm_wiki.storage.metadata as metadata_module
from llm_wiki.storage.metadata import LLMCallLog, log_llm_call
from tests.conftest import TEST_DATABASE_URL


@pytest.fixture(autouse=True)
def _sync_engine_on_test_db(db_engine) -> Iterator[None]:
    """Point log_llm_call's lazy sync engine at the per-test database.

    Without this it would write into settings.database_url — the LIVE
    database — which is exactly the kind of accident these tests guard
    against.
    """
    engine = create_engine(TEST_DATABASE_URL)
    old = metadata_module._sync_engine
    metadata_module._sync_engine = engine
    yield
    metadata_module._sync_engine = old
    engine.dispose()


@pytest.mark.asyncio
async def test_log_ok_call_persists_row(db_session: AsyncSession) -> None:
    log_llm_call(
        file_id="f1",
        agent_type="advisor",
        model="gpt-5.4",
        status="ok",
        duration_ms=1234,
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.001,
        attempts=1,
    )

    rows = (await db_session.execute(select(LLMCallLog))).scalars().all()
    assert len(rows) == 1
    row = rows[0]
    assert row.status == "ok"
    assert row.duration_ms == 1234
    assert row.error_type == ""
    assert row.timestamp is not None


@pytest.mark.asyncio
async def test_log_error_call_records_exception(db_session: AsyncSession) -> None:
    log_llm_call(
        file_id="f2",
        agent_type="writer",
        model="gpt-5.4",
        status="error",
        duration_ms=21000,
        attempts=3,
        error=TimeoutError("read timed out" + "x" * 600),
    )

    rows = (await db_session.execute(select(LLMCallLog))).scalars().all()
    assert len(rows) == 1
    row = rows[0]
    assert row.status == "error"
    assert row.attempts == 3
    assert row.error_type == "TimeoutError"
    assert len(row.error_message) == 500  # truncated


@pytest.mark.asyncio
async def test_write_failure_never_raises(db_session: AsyncSession) -> None:
    with patch(
        "llm_wiki.storage.metadata.get_sync_engine",
        side_effect=RuntimeError("db down"),
    ):
        # Must swallow the failure — telemetry can't break the LLM call.
        log_llm_call(
            file_id="f3",
            agent_type="advisor",
            model="gpt-5.4",
            status="ok",
            duration_ms=1,
        )

    rows = (await db_session.execute(select(LLMCallLog))).scalars().all()
    assert rows == []
