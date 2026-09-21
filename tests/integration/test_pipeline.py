"""Full ingestion against PostgreSQL/pgvector and a temporary object store.

Only model calls are stubbed. Wiki bodies, headings, chunks, state transitions
and notifications use the same stores as the application.
"""

import json
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from sqlalchemy import select
from sqlalchemy.exc import DBAPIError

from llm_wiki.orchestrator import pipeline
from llm_wiki.storage import wiki_store
from llm_wiki.storage.index import IndexStorage
from llm_wiki.storage.metadata import FileRecord, NotificationRecord
from llm_wiki.storage.object_store import get_object_store

FILE_ID = "pipeline-test"
BODY = "# Transformers\n\n" + (
    "Self-attention layers and positional embeddings encode sequences. "
    "Encoder-decoder stacks support translation and summarization. "
) * 3


@pytest.fixture
async def ingestion(db_engine, vector_engine, db_session, monkeypatch):
    monkeypatch.setattr("llm_wiki.api.deps._engine", db_engine)
    monkeypatch.setattr("llm_wiki.storage.metadata._sync_engine", vector_engine)
    llm = MagicMock()
    llm.load_prompt.return_value = "prompt"
    llm.complete = AsyncMock(return_value=(json.dumps({"slug": "transformers", "title": "Transformers", "content": BODY}), None))
    llm.embed.side_effect = lambda texts, **_kw: [[0.1] * 1536 for _ in texts]
    llm.aclose = AsyncMock()
    client = Mock(return_value=llm)
    monkeypatch.setattr(pipeline, "LLMClient", client)
    get_object_store().put_text(f"raw/{FILE_ID}.md", BODY)
    db_session.add(FileRecord(file_id=FILE_ID, original_name="architecture.md", raw_key=f"raw/{FILE_ID}.md", status="RECEIVED", owner="pipeline@test", sensitive=False))
    await db_session.commit()
    return llm, client


async def test_upload_md_creates_wiki_page(ingestion, db_session):
    from llm_wiki.llm.chunk_store import ChunkStore

    await pipeline.process_file(FILE_ID)
    assert wiki_store.get_page("transformers") == BODY.strip()
    assert any(p[0] == "transformers" for p in IndexStorage().read_pages())
    assert ChunkStore(llm_client=ingestion[0]).count() > 0
    record = await db_session.get(FileRecord, FILE_ID, populate_existing=True)
    assert record.status == "DONE"
    assert [e["state"] for e in record.state_history] == ["STORED", "SEARCHED", "WRITTEN", "LINTED", "LOGGED", "DONE"]
    assert record.finished_at is not None
    assert record.created_pages == ["transformers"]
    await db_session.refresh(record, attribute_names=["extracted_text"])
    assert record.extracted_text.startswith("Transformers\n")
    assert record.extracted_text.count("Self-attention layers") == 3
    assert len(list(await db_session.scalars(select(NotificationRecord)))) == 1


async def test_pipeline_idempotent_on_rerun(ingestion, db_session):
    llm, client = ingestion
    await pipeline.process_file(FILE_ID)
    record = await db_session.get(FileRecord, FILE_ID, populate_existing=True)
    before = (record.finished_at, record.updated_at, record.state_history)
    event = (await db_session.scalars(select(NotificationRecord))).one()
    event_time = event.created_at
    completion_calls, embedding_calls = llm.complete.call_count, llm.embed.call_count
    await pipeline.process_file(FILE_ID)
    await db_session.refresh(record)
    await db_session.refresh(event)
    assert (record.finished_at, record.updated_at, record.state_history) == before
    assert event.created_at == event_time
    assert llm.complete.call_count == completion_calls
    assert llm.embed.call_count == embedding_calls
    assert client.call_count == 1
    assert len(wiki_store.list_pages()) == 1


async def test_pipeline_failed_state_on_llm_error(ingestion, db_session):
    llm, _client = ingestion
    llm.complete.side_effect = RuntimeError("LLM exploded")
    with pytest.raises(RuntimeError, match="LLM exploded"):
        await pipeline.process_file(FILE_ID)
    record = await db_session.get(FileRecord, FILE_ID, populate_existing=True)
    assert record.status == "FAILED"
    assert record.error
    assert not wiki_store.list_pages()
    llm.aclose.assert_awaited_once()
    assert len(list(await db_session.scalars(select(NotificationRecord)))) == 1


async def test_ingestion_cleans_nul_from_parser_before_caching(ingestion, db_session, monkeypatch):
    text = "Начало\x00 текста\nҚазақша\tEnglish 🙂"
    monkeypatch.setattr(pipeline, "_load_raw_text", Mock(return_value=text))
    await pipeline.process_file(FILE_ID)
    record = await db_session.get(FileRecord, FILE_ID, populate_existing=True)
    await db_session.refresh(record, attribute_names=["extracted_text"])
    assert record.status == "DONE"
    assert record.extracted_text == "Начало текста\nҚазақша\tEnglish 🙂"


async def test_ingestion_records_failure_after_database_rejects_a_write(ingestion, db_session, monkeypatch):
    async def fail_transition(session, file_id, _state):
        record = await session.get(FileRecord, file_id)
        record.extracted_text = "Invalid\x00text"
        await session.commit()

    monkeypatch.setattr(pipeline, "_transition", fail_transition)
    with pytest.raises(DBAPIError):
        await pipeline.process_file(FILE_ID)
    record = await db_session.get(FileRecord, FILE_ID, populate_existing=True)
    assert record.status == "FAILED"
    assert record.finished_at is not None
    assert "сохранить данные" in record.error
    event = (await db_session.scalars(select(NotificationRecord))).one()
    assert event.event == "failed"
    ingestion[0].aclose.assert_awaited_once()


async def test_ingestion_records_client_initialization_failure(ingestion, db_session, monkeypatch):
    monkeypatch.setattr(pipeline, "LLMClient", Mock(side_effect=RuntimeError("client init failed")))
    with pytest.raises(RuntimeError, match="client init failed"):
        await pipeline.process_file(FILE_ID)
    record = await db_session.get(FileRecord, FILE_ID, populate_existing=True)
    assert record.status == "FAILED"
    assert record.finished_at is not None
    assert (await db_session.scalars(select(NotificationRecord))).one().event == "failed"


async def test_old_task_does_not_republish_rolled_back_source(ingestion, db_session):
    record = await db_session.get(FileRecord, FILE_ID)
    record.status = "ROLLED_BACK"
    await db_session.commit()
    await pipeline.process_file(FILE_ID)
    ingestion[1].assert_not_called()
    await db_session.refresh(record)
    assert record.status == "ROLLED_BACK"
    assert not wiki_store.list_pages()


async def test_retry_after_writing_does_not_repeat_search_or_writer(ingestion, db_session, monkeypatch):
    from llm_wiki.storage.metadata import update_file_status

    await pipeline.process_file(FILE_ID)
    record = await db_session.get(FileRecord, FILE_ID, populate_existing=True)
    record.state_history = [e for e in record.state_history if e["state"] in {"STORED", "SEARCHED", "WRITTEN"}]
    await db_session.commit()
    await update_file_status(db_session, FILE_ID, "FAILED")
    llm = ingestion[0]
    llm.reset_mock()
    search = AsyncMock(side_effect=AssertionError("Search must not be repeated"))
    monkeypatch.setattr(pipeline.SearchAgent, "run", search)
    await pipeline.process_file(FILE_ID)
    search.assert_not_called()
    llm.complete.assert_not_called()
    llm.embed.assert_not_called()
    await db_session.refresh(record)
    assert record.status == "DONE"
    assert wiki_store.get_page("transformers") == BODY.strip()


async def test_redelivery_recovers_event_after_crash_between_done_and_notification(ingestion, db_session, monkeypatch):
    with monkeypatch.context() as first_run:
        first_run.setattr("llm_wiki.storage.notifications.notify_file_done", AsyncMock())
        await pipeline.process_file(FILE_ID)
    record = await db_session.get(FileRecord, FILE_ID, populate_existing=True)
    finished_at = record.finished_at
    assert not list(await db_session.scalars(select(NotificationRecord)))
    await pipeline.process_file(FILE_ID)
    event = (await db_session.scalars(select(NotificationRecord))).one()
    assert event.occurred_at == finished_at
    ingestion[1].assert_called_once()
