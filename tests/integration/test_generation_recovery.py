"""Production NUL/failed-transaction regressions against real PostgreSQL.

No provider calls: only parsing/model outputs are stubbed. Task state changes,
source caching, result writes and notifications use the real database.
"""

import asyncio
import json
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError

from llm_wiki.config import settings
from llm_wiki.orchestrator import tasks
from llm_wiki.storage.metadata import ArtifactRecord, FileRecord, NotificationRecord
from llm_wiki.storage.object_store import get_object_store

SOURCE = "nul-source"
ARTIFACT = "nul-artifact"
ATTEMPT = "attempt-1"
TEXT = "Начало\x00 текста\nҚазақша\tEnglish\x00 🙂"  # noqa: RUF001 — multilingual fixture
CLEAN_TEXT = "Начало текста\nҚазақша\tEnglish 🙂"  # noqa: RUF001 — multilingual fixture
OLD_VERSIONS = [{"language": "ru", "content": {"summary": "Previous result"}}]


@pytest.fixture
async def generation(db_engine, db_session, monkeypatch):
    monkeypatch.setattr(settings, "database_url", db_engine.url.render_as_string(hide_password=False))
    db_session.add_all([
        FileRecord(file_id=SOURCE, original_name="source.txt", status="DONE"),
        ArtifactRecord(
            artifact_id=ARTIFACT, document_id=SOURCE, kind="report", status="pending",
            versions=OLD_VERSIONS, requested_by="owner@test",
            generation_context={"id": ATTEMPT, "source_doc_ids": [SOURCE]},
        ),
    ])
    await db_session.commit()
    get_object_store().put_text(f"raw/{SOURCE}.txt", TEXT)
    llm = Mock()
    llm.complete = AsyncMock(return_value=(json.dumps({"summary": "New result"}), None))
    llm.aclose = AsyncMock()
    monkeypatch.setattr("llm_wiki.llm.client.LLMClient", Mock(return_value=llm))
    return llm


async def run_generation():
    return await asyncio.to_thread(
        tasks.generate_artifact.run, ARTIFACT, SOURCE, "report", "ru", ATTEMPT,
    )


@pytest.mark.parametrize("extension", ["txt", "pdf"])
async def test_nul_source_generates_and_caches_clean_text(
    generation, db_session, monkeypatch, extension,
):
    source = await db_session.get(FileRecord, SOURCE)
    source.raw_key = f"raw/{SOURCE}.{extension}"
    await db_session.commit()
    get_object_store().put_text(source.raw_key, TEXT)
    if extension == "pdf":
        monkeypatch.setattr("llm_wiki.orchestrator.pipeline.parse_pdf", Mock(return_value=TEXT))
        monkeypatch.setattr(settings, "ocr_enabled", False)

    assert (await run_generation())["status"] == "ready"
    await db_session.refresh(source, attribute_names=["extracted_text"])
    assert source.extracted_text == CLEAN_TEXT
    prompt_args = generation.load_prompt.call_args.kwargs
    assert CLEAN_TEXT in prompt_args["content"]
    artifact = await db_session.get(ArtifactRecord, ARTIFACT, populate_existing=True)
    assert artifact.versions[0]["content"]["summary"] == "New result"
    event = (await db_session.scalars(select(NotificationRecord))).one()
    assert event.event == "done"
    generation.aclose.assert_awaited_once()


async def poison_source_transaction(session, _llm=None, **_kwargs):
    source = await session.get(FileRecord, SOURCE)
    source.extracted_text = "Invalid\x00source"
    await session.commit()  # real PostgreSQL DataError, leaving a failed session


@pytest.mark.parametrize("failure", ["source_write", "result_write", "client_init", "timeout"])
async def test_failure_is_persisted_immediately_and_preserves_previous_result(
    generation, db_session, monkeypatch, failure,
):
    if failure == "source_write":
        monkeypatch.setattr("llm_wiki.agents.artifacts.generate_content", poison_source_transaction)
    elif failure == "result_write":
        # Force the real result UPDATE to fail while allowing the subsequent
        # failed-state UPDATE. This exercises transaction recovery, not a mock.
        await db_session.execute(text(
            "ALTER TABLE artifacts ADD CONSTRAINT reject_ready_for_test CHECK (status <> 'ready')"
        ))
        await db_session.commit()
    elif failure == "client_init":
        monkeypatch.setattr("llm_wiki.llm.client.LLMClient", Mock(side_effect=RuntimeError("client init failed")))
    else:
        monkeypatch.setattr("llm_wiki.agents.artifacts.generate_content", AsyncMock(side_effect=TimeoutError))

    assert (await run_generation())["status"] == "failed"
    artifact = await db_session.get(ArtifactRecord, ARTIFACT, populate_existing=True)
    assert artifact.status == "failed"
    assert artifact.finished_at is not None
    assert artifact.versions == OLD_VERSIONS
    assert artifact.error
    assert "PendingRollbackError" not in artifact.error
    assert "UPDATE " not in artifact.error
    if failure in {"source_write", "result_write"}:
        assert "Ошибка работы с данными" in artifact.error  # noqa: RUF001
    event = (await db_session.scalars(select(NotificationRecord))).one()
    assert (event.event, event.entity_id, event.recipient) == ("failed", ARTIFACT, "owner@test")
    assert event.detail == artifact.error
    if failure != "client_init":
        generation.aclose.assert_awaited_once()


async def test_failed_old_transaction_cannot_cancel_a_new_attempt(
    generation, db_session, monkeypatch,
):
    async def fail_and_replace(session, llm, **kwargs):
        try:
            await poison_source_transaction(session, llm, **kwargs)
        except DBAPIError:
            # Simulate a new generation queued while the failed task unwinds.
            async with tasks._worker_session_factory()() as replacement:
                row = await replacement.get(ArtifactRecord, ARTIFACT)
                row.generation_context = {"id": "attempt-2", "source_doc_ids": [SOURCE]}
                row.started_at = None
                await replacement.commit()
            raise

    monkeypatch.setattr("llm_wiki.agents.artifacts.generate_content", fail_and_replace)
    assert (await run_generation())["status"] == "skipped"
    artifact = await db_session.get(ArtifactRecord, ARTIFACT, populate_existing=True)
    assert artifact.status == "pending"
    assert artifact.generation_context["id"] == "attempt-2"
    assert artifact.versions == OLD_VERSIONS
    assert artifact.error is None
    assert not list(await db_session.scalars(select(NotificationRecord)))
