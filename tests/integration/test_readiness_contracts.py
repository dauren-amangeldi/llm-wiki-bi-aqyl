"""HTTP + PostgreSQL regressions; no external jobs or paid model calls."""
import asyncio
from unittest.mock import Mock

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker

from llm_wiki.api.deps import get_db
from llm_wiki.main import app
from llm_wiki.storage.metadata import AdvisorConsultation, ArtifactRecord, CaseRecord, FileRecord, NotificationRecord
from llm_wiki.storage import artifacts_store
from sqlalchemy import select

OWNER = "readiness@bi.group"

@pytest_asyncio.fixture
async def client(db_engine, monkeypatch):
    factory = async_sessionmaker(db_engine, expire_on_commit=False)
    async def database():
        async with factory() as session:
            yield session
    app.dependency_overrides[get_db] = database
    monkeypatch.setattr("llm_wiki.api.v1.cases._dispatch_autotag", Mock())
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", headers={"X-User-Email": OWNER}) as http:
            yield http
    finally:
        app.dependency_overrides.clear()


async def test_concurrent_membership_edits_do_not_resurrect_removed_source(client, db_session):
    for fid in ("a", "b", "c"):
        db_session.add(FileRecord(file_id=fid, original_name=fid + ".md", status="DONE", sensitive=True, owner=OWNER))
    db_session.add(CaseRecord(id="case", title="Case", doc_ids=["a", "b"], sensitive=True, owner=OWNER, description="outdated"))
    await db_session.commit()
    removed, added = await asyncio.gather(
        client.delete("/api/v1/cases/case/documents/a"),
        client.put("/api/v1/cases/case/documents/c"),
    )
    assert removed.status_code == added.status_code == 200
    row = await db_session.get(CaseRecord, "case")
    assert set(row.doc_ids) == {"b", "c"}
    assert row.description == ""
    assert (await client.put("/api/v1/cases/case/documents/c")).status_code == 200
    await db_session.refresh(row)
    assert row.doc_ids.count("c") == 1
    assert list(await db_session.scalars(select(NotificationRecord)))
    assert (await client.put("/api/v1/cases/case/documents/a", headers={"X-User-Email": "other@bi.group"})).status_code == 403


async def test_unlink_last_public_membership_restores_private_visibility(client, db_session):
    db_session.add(FileRecord(file_id="shared", original_name="shared.md", status="DONE", sensitive=False, owner=OWNER))
    db_session.add_all([
        CaseRecord(id="public", title="Public", doc_ids=["shared"], sensitive=False, owner=OWNER),
        CaseRecord(id="private", title="Private", doc_ids=["shared"], sensitive=True, owner=OWNER),
    ])
    await db_session.commit()
    assert (await client.delete("/api/v1/cases/public/documents/shared")).status_code == 200
    source = await db_session.get(FileRecord, "shared")
    assert source.sensitive is True
    assert (await client.get("/api/v1/documents/shared", headers={"X-User-Email": "other@bi.group"})).status_code == 404


async def test_status_batch_includes_rolled_back_but_never_foreign_private_files(client, db_session):
    db_session.add_all([
        FileRecord(file_id="rolled", original_name="a.md", status="ROLLED_BACK", sensitive=True, owner=OWNER),
        FileRecord(file_id="hidden", original_name="b.md", status="DONE", sensitive=True, owner="other@bi.group"),
        FileRecord(file_id="unrequested", original_name="c.md", status="DONE"),
    ])
    await db_session.commit()
    rows = (await client.get("/api/v1/documents?ids=rolled&ids=hidden")).json()
    assert [(r["document_id"], r["status"]) for r in rows] == [("rolled", "ROLLED_BACK")]
    assert "rolled" not in [r["document_id"] for r in (await client.get("/api/v1/documents")).json()]


async def test_history_pages_complete_briefs_for_the_current_account(client, db_session):
    for i in range(4):
        db_session.add(AdvisorConsultation(id=str(i), owner=OWNER if i < 3 else "other@bi.group", title=str(i), situation="s", step="recommendation" if i else "questions", brief={"headline": str(i)} if i else None))
    await db_session.commit()
    endpoint = "/api/v1/advisor/consultations?include_brief=true&completed_only=true&limit=1"
    first = (await client.get(endpoint)).json()
    second = (await client.get(endpoint + "&offset=1")).json()
    assert {first[0]["id"], second[0]["id"]} == {"1", "2"}
    assert first[0]["brief"]["headline"] == first[0]["id"]
    assert (await client.get(endpoint + "&offset=2")).json() == []


async def test_old_failure_cannot_finish_new_artifact_attempt(db_session):
    row = ArtifactRecord(artifact_id="a", document_id="d", kind="report", status="pending", generation_context={"id": "new"})
    db_session.add(row)
    await db_session.commit()
    assert await artifacts_store.mark_failed(db_session, "a", "old error", generation_id="old") is False
    await db_session.refresh(row)
    assert row.status == "pending"
    assert await artifacts_store.mark_failed(db_session, "a", "new error", generation_id="new") is True


async def test_shared_wiki_does_not_mix_unselected_originals_and_legacy_parse_is_cached(db_session, monkeypatch):
    from llm_wiki.agents.artifacts import _load_selected_sources
    db_session.add_all([
        FileRecord(file_id="a", original_name="a.md", status="DONE", created_pages=["shared"]),
        FileRecord(file_id="b", original_name="b.md", status="DONE", updated_pages=["shared"], extracted_text="UNSELECTED CONFIDENTIAL"),
    ])
    await db_session.commit()
    parse = Mock(return_value="SELECTED ORIGINAL")
    monkeypatch.setattr("llm_wiki.orchestrator.pipeline._load_raw_text", parse)
    for _ in range(2):
        body, titles = await _load_selected_sources(db_session, ["a"])
        assert "SELECTED ORIGINAL" in body
        assert "UNSELECTED" not in body
        assert titles == ["a.md"]
    parse.assert_called_once()


async def test_ingestion_retry_reuses_parsed_source_after_later_stage_failure(db_engine, db_session, monkeypatch):
    from unittest.mock import AsyncMock, MagicMock
    from llm_wiki.orchestrator.pipeline import process_file
    db_session.add(FileRecord(file_id="retry", original_name="scan.pdf", status="RECEIVED", sensitive=True, owner=OWNER))
    await db_session.commit()
    llm = MagicMock()
    llm.aclose = AsyncMock()
    monkeypatch.setattr("llm_wiki.orchestrator.pipeline.LLMClient", lambda: llm)
    monkeypatch.setattr("llm_wiki.api.deps._engine", db_engine)
    parse = Mock(return_value="Original recognized text")
    monkeypatch.setattr("llm_wiki.orchestrator.pipeline._load_raw_text", parse)
    monkeypatch.setattr("llm_wiki.orchestrator.pipeline.IndexStorage", Mock(side_effect=RuntimeError("later stage failed")))
    import pytest
    for _ in range(2):
        with pytest.raises(RuntimeError, match="later stage failed"):
            await process_file("retry")
    parse.assert_called_once()
    row = await db_session.get(FileRecord, "retry", populate_existing=True)
    await db_session.refresh(row, attribute_names=["extracted_text"])
    assert row.extracted_text == "Original recognized text"
