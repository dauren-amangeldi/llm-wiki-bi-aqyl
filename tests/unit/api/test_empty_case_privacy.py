"""Incomplete cases stay private through the complete material lifecycle."""

from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker

from llm_wiki.api.deps import get_db
from llm_wiki.api.v1 import cases, twins
from llm_wiki.main import app
from llm_wiki.storage import notifications, wiki_store
from llm_wiki.storage.case_visibility import privatize_unready_cases
from llm_wiki.storage.metadata import (
    CaseRecord,
    ChunkEmbedding,
    FileRecord,
    _EMBED_DIM,
    append_twin_message,
    create_twin_session,
    update_file_status,
)

OWNER = {"X-User-Email": "alice"}
OTHER = {"X-User-Email": "bob"}


@pytest.fixture
async def client(db_engine, monkeypatch):
    factory = async_sessionmaker(db_engine, expire_on_commit=False)

    async def database():
        async with factory() as db:
            yield db

    app.dependency_overrides[get_db] = database
    monkeypatch.setattr(cases, "_dispatch_autotag", lambda _: None)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.pop(get_db, None)


async def add_file(db, fid="ready", status="DONE"):
    db.add(
        FileRecord(
            file_id=fid,
            original_name=f"{fid}.md",
            status=status,
            owner="alice",
            sensitive=True,
            created_pages=[f"page-{fid}"],
        )
    )
    await db.commit()
    wiki_store.save_page(
        f"page-{fid}", "Knowledge", "Knowledge from a case", sensitive=True, owner="alice"
    )


async def create_case(client, ids=(), sensitive=True):
    response = await client.post(
        "/api/v1/cases",
        headers=OWNER,
        json={
            "id": "case",
            "title": "Case",
            "doc_ids": list(ids),
            "sensitive": sensitive,
        },
    )
    assert response.status_code == 201
    return response.json()


async def visible(client, headers):
    response = await client.get("/api/v1/cases", headers=headers)
    assert response.status_code == 200
    return response.json()


@pytest.mark.parametrize("status", [None, "STORED", "FAILED", "ROLLED_BACK"])
async def test_incomplete_case_is_private_and_cannot_be_published(client, db_session, status):
    if status:
        await add_file(db_session, status=status)
    saved = await create_case(client, ["ready"] if status else [], sensitive=False)
    assert saved["sensitive"] is True and saved["owner"] == "alice"
    assert len(await visible(client, OWNER)) == 1
    assert await visible(client, OTHER) == []
    assert (await client.get("/api/v1/twin/readiness", headers=OTHER)).json() == {}
    assert (await client.get("/api/v1/cases?category=public", headers=OWNER)).json() == []
    published = await client.put(
        "/api/v1/cases/case", headers=OWNER, json={"title": "Case", "sensitive": False}
    )
    assert published.status_code == 422
    # Rename is allowed without accidentally publishing via the old default.
    renamed = await client.put("/api/v1/cases/case", headers=OWNER, json={"title": "Renamed"})
    assert renamed.status_code == 200
    assert (await visible(client, OWNER))[0]["title"] == "Renamed"


async def test_first_done_unlocks_but_requires_explicit_publication(client, db_session):
    await create_case(client, sensitive=False)
    await add_file(db_session, status="STORED")
    assert (
        await client.put("/api/v1/cases/case/documents/ready", headers=OWNER)
    ).status_code == 200
    await update_file_status(db_session, "ready", "DONE")
    saved = (await visible(client, OWNER))[0]
    assert saved["sensitive"] is True
    assert saved["council"]["ready_doc_ids"] == ["ready"]
    assert await visible(client, OTHER) == []
    assert (await client.get("/api/v1/wiki?q=knowledge", headers=OTHER)).json() == []
    published = await client.put(
        "/api/v1/cases/case", headers=OWNER, json={"title": "Case", "sensitive": False}
    )
    assert published.status_code == 200
    assert len(await visible(client, OTHER)) == 1
    assert len((await client.get("/api/v1/wiki?q=knowledge", headers=OTHER)).json()) == 1


@pytest.mark.parametrize("removal", ["unlink", "replace", "rollback"])
async def test_last_ready_removal_demotes_and_preserves_history(
    client, db_session, monkeypatch, removal
):
    await add_file(db_session)
    await add_file(db_session, "pending", "STORED")
    await create_case(client, ["ready", "pending"], sensitive=False)
    db_session.add(
        ChunkEmbedding(
            id="pending#0",
            file_id="pending",
            slug="page-pending",
            document="Knowledge",
            embedding=[0.1] * _EMBED_DIM,
        )
    )
    await db_session.commit()
    session = await create_twin_session(
        db_session, case_id="case", persona_ids=[], created_by="alice"
    )
    sid = session.id
    await append_twin_message(
        db_session,
        session_id=sid,
        role="persona",
        persona_id=None,
        seq=0,
        content={"text": "Saved opinion"},
    )
    await notifications.notify_case_ready_if_done(db_session, "case")
    await notifications.upsert_event(
        db_session,
        section="cases",
        family="privacy",
        event="published",
        entity_id="case",
        title="Case",
    )
    assert len(await notifications.list_events(db_session, "bob")) >= 1
    if removal == "unlink":
        response = await client.delete("/api/v1/cases/case/documents/ready", headers=OWNER)
    elif removal == "replace":
        response = await client.put(
            "/api/v1/cases/case",
            headers=OWNER,
            json={"title": "Case", "doc_ids": ["pending"], "sensitive": False},
        )
    else:
        response = await client.delete("/api/v1/documents/ready", headers=OWNER)
    assert response.status_code == 200
    assert await visible(client, OTHER) == []
    saved = (await visible(client, OWNER))[0]
    assert saved["sensitive"] is True and saved["council"]["ready_doc_ids"] == []
    assert (await client.get("/api/v1/wiki?q=knowledge", headers=OTHER)).json() == []
    assert await notifications.list_events(db_session, "bob") == []
    db_session.expire_all()
    assert (await db_session.get(FileRecord, "pending")).sensitive is True
    chunk = await db_session.scalar(
        select(ChunkEmbedding).where(ChunkEmbedding.file_id == "pending")
    )
    assert chunk.sensitive is True and chunk.owner == "alice"
    assert (
        await db_session.execute(text("SELECT sensitive FROM wiki_fts WHERE slug='page-pending'"))
    ).scalar() is True
    # Session history is stored unchanged even while all inference is blocked.
    from llm_wiki.storage.metadata import TwinMessage

    assert (
        await db_session.scalar(select(TwinMessage).where(TwinMessage.session_id == sid))
    ).content["text"] == "Saved opinion"
    constructor = MagicMock()
    monkeypatch.setattr("llm_wiki.llm.client.LLMClient", constructor)
    from fastapi import HTTPException

    with pytest.raises(HTTPException):
        await twins.twin_chat_endpoint(
            twins.TwinChatRequest(
                case_id="case", session_id=sid, message="Continue", persona_ids=["jobs"]
            ),
            db_session,
            "alice",
        )
    constructor.assert_not_called()
    # Completing the remaining source restores readiness, never public access.
    await update_file_status(db_session, "pending", "DONE")
    assert (await visible(client, OWNER))[0]["sensitive"] is True
    assert await visible(client, OTHER) == []


async def test_shared_source_stays_public_if_another_ready_case_uses_it(client, db_session):
    await add_file(db_session)
    await create_case(client, ["ready"], sensitive=False)
    db_session.add(
        CaseRecord(id="other", title="Other", owner="alice", sensitive=False, doc_ids=["ready"])
    )
    await db_session.commit()
    await client.delete("/api/v1/cases/case/documents/ready", headers=OWNER)
    assert [c["id"] for c in await visible(client, OTHER)] == ["other"]
    db_session.expire_all()
    assert (await db_session.get(FileRecord, "ready")).sensitive is False
    assert len((await client.get("/api/v1/wiki?q=knowledge", headers=OTHER)).json()) == 1


async def test_legacy_demotion_is_idempotent_and_first_attachment_does_not_republish(
    client, db_session
):
    await add_file(db_session, "pending", "STORED")
    db_session.add(
        CaseRecord(id="case", title="Legacy", owner="alice", sensitive=False, doc_ids=["pending"])
    )
    await db_session.commit()
    assert await privatize_unready_cases(db_session) == 1
    await db_session.commit()
    assert await privatize_unready_cases(db_session) == 0
    await add_file(db_session)
    await client.put(
        "/api/v1/cases/case", headers=OWNER, json={"title": "Case", "doc_ids": ["ready"]}
    )
    assert (await visible(client, OWNER))[0]["sensitive"] is True
    assert await visible(client, OTHER) == []


async def test_case_membership_cannot_take_another_owners_private_file(client, db_session):
    db_session.add(
        FileRecord(
            file_id="secret", original_name="secret.md", status="DONE", sensitive=True, owner="bob"
        )
    )
    await db_session.commit()
    attempted = await client.post(
        "/api/v1/cases", headers=OWNER, json={"title": "Forged", "doc_ids": ["secret"]}
    )
    assert attempted.status_code == 404
    await create_case(client)
    attempted = await client.put(
        "/api/v1/cases/case", headers=OWNER, json={"title": "Case", "doc_ids": ["secret"]}
    )
    assert attempted.status_code == 404
    db_session.expire_all()
    secret = await db_session.get(FileRecord, "secret")
    assert secret.sensitive is True and secret.owner == "bob"
    assert (await client.get("/api/v1/documents/secret/sources", headers=OWNER)).json() == []
    assert (await client.get("/api/v1/documents/secret/related", headers=OWNER)).json() == {
        "items": []
    }


async def test_unlinking_nonmember_cannot_change_another_files_visibility(client, db_session):
    await create_case(client)
    db_session.add(
        FileRecord(file_id="unrelated", original_name="public.md", status="DONE", sensitive=False)
    )
    await db_session.commit()
    response = await client.delete("/api/v1/cases/case/documents/unrelated", headers=OWNER)
    assert response.status_code == 200
    db_session.expire_all()
    assert (await db_session.get(FileRecord, "unrelated")).sensitive is False
