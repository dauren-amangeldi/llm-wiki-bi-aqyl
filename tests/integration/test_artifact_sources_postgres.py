"""Real PostgreSQL and HTTP contract; queue is stubbed, no paid LLM calls."""
import asyncio
from unittest.mock import Mock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker

from llm_wiki.api.deps import get_db
from llm_wiki.main import app
from llm_wiki.agents.artifacts import _title_and_slugs
from llm_wiki.storage import artifacts_store
from llm_wiki.storage.metadata import ArtifactRecord, CaseRecord, FileRecord

OWNER = "owner@bi.group"


@pytest_asyncio.fixture
async def client(db_engine, db_session, monkeypatch):
    factory = async_sessionmaker(db_engine, expire_on_commit=False)
    async def database():
        async with factory() as session:
            yield session
    app.dependency_overrides[get_db] = database
    for name in ("a", "b"):
        db_session.add(FileRecord(file_id=name, original_name=f"{name}.md", status="DONE", created_pages=[f"page-{name}"], sensitive=True, owner=OWNER))
    db_session.add(CaseRecord(id="case-selection", title="Selection", doc_ids=["a", "b"], sensitive=True, owner=OWNER))
    await db_session.commit()
    from llm_wiki.orchestrator.tasks import generate_artifact
    enqueue = Mock()
    monkeypatch.setattr(generate_artifact, "apply_async", enqueue)
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test", headers={"X-User-Email": OWNER}) as http:
            yield http, enqueue
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint,extra", [("studio", {"kind": "report", "language": "ru"}), ("cards", {"languages": ["ru"]}), ("images", {"language": "ru"})])
async def test_selection_is_queued_and_excludes_unselected_pages(client, db_session, endpoint, extra):
    http, enqueue = client
    response = await http.post(f"/api/v1/{endpoint}/generate", json={"document_id": "case-selection", "source_doc_ids": ["a"], **extra})
    assert response.status_code in {200, 202}
    record = await db_session.get(ArtifactRecord, response.json()["artifact_id"])
    assert record.generation_context["source_doc_ids"] == ["a"]
    assert enqueue.call_args.kwargs["kwargs"] == {"generation_id": record.generation_context["id"]}
    assert (await _title_and_slugs(db_session, "case-selection", record.generation_context["source_doc_ids"]))[1] == ["page-a"]


@pytest.mark.asyncio
@pytest.mark.parametrize("selection", [[], ["foreign"], ["a", 42]])
async def test_bad_selection_does_not_replace_ready_version(client, db_session, selection):
    http, enqueue = client
    old = await artifacts_store.upsert_artifact(db_session, document_id="case-selection", kind="report", language="ru", content={"summary": "original"}, source_doc_ids=["b"])
    response = await http.post("/api/v1/studio/generate", json={"document_id": "case-selection", "kind": "report", "source_doc_ids": selection})
    assert response.status_code == 422
    enqueue.assert_not_called()
    await db_session.refresh(old)
    assert old.status == "ready"
    assert old.versions[0]["source_doc_ids"] == ["b"]


@pytest.mark.asyncio
async def test_two_concurrent_starts_enqueue_once_and_reject_different_context(client):
    http, enqueue = client
    body = {"document_id": "case-selection", "kind": "report", "source_doc_ids": ["a"]}
    first, second = await asyncio.gather(http.post("/api/v1/studio/generate", json=body), http.post("/api/v1/studio/generate", json=body))
    assert first.status_code == second.status_code == 202
    assert first.json()["artifact_id"] == second.json()["artifact_id"]
    enqueue.assert_called_once()
    different = await http.post("/api/v1/studio/generate", json={**body, "source_doc_ids": ["b"]})
    assert different.status_code == 409
    enqueue.assert_called_once()


@pytest.mark.asyncio
async def test_failed_regeneration_exposes_previous_success_and_its_sources(client, db_session):
    http, _ = client
    old = await artifacts_store.upsert_artifact(db_session, document_id="case-selection", kind="report", language="ru", content={"summary": "original"}, source_doc_ids=["b"])
    await http.post("/api/v1/studio/generate", json={"document_id": "case-selection", "kind": "report", "source_doc_ids": ["a"]})
    await artifacts_store.mark_failed(db_session, old.artifact_id, "failed attempt")
    response = await http.get("/api/v1/artifacts?document_id=case-selection&language=ru")
    assert response.json()[0]["has_content"] is True
    assert response.json()[0]["source_doc_ids"] == ["b"]
    assert response.json()[0]["status"] == "failed"


@pytest.mark.asyncio
async def test_private_artifact_and_exports_are_not_readable_by_another_account(client, db_session):
    http, enqueue = client
    old = await artifacts_store.upsert_artifact(db_session, document_id="case-selection", kind="report", language="ru", content={"summary": "private"})
    headers = {"X-User-Email": "other@bi.group"}
    for url in ("/api/v1/artifacts?document_id=case-selection", f"/api/v1/artifacts/{old.artifact_id}", f"/api/v1/artifacts/{old.artifact_id}/export?format=pdf"):
        assert (await http.get(url, headers=headers)).status_code == 404
    assert (await http.post("/api/v1/studio/generate", headers=headers, json={"document_id": "case-selection", "kind": "report"})).status_code == 404
    enqueue.assert_not_called()


async def test_broker_failure_keeps_previous_content_and_never_runs_inline(client, db_session):
    http, enqueue = client
    old = await artifacts_store.upsert_artifact(db_session, document_id="case-selection", kind="report", language="ru", content={"summary": "original"}, source_doc_ids=["b"])
    enqueue.side_effect = ConnectionError("broker down")
    response = await http.post("/api/v1/studio/generate", json={"document_id": "case-selection", "kind": "report", "source_doc_ids": ["a"]})
    assert response.status_code == 503
    await db_session.refresh(old)
    assert old.status == "failed"
    assert old.versions[0]["content"] == {"summary": "original"}
