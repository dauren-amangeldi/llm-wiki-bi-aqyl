from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from llm_wiki.api.deps import get_db, get_user_key
from llm_wiki.main import app
from llm_wiki.storage.metadata import CaseRecord, FileRecord, create_twin_session, append_twin_message
from llm_wiki.storage.wiki_fts import upsert_wiki_fts


async def test_reopened_council_resolves_real_sources_and_hides_inaccessible_ones(db_session):
    owner = "alice@bi.group"
    db_session.add(CaseRecord(id="case", title="Case", doc_ids=["file"], owner=owner))
    db_session.add(FileRecord(file_id="file", original_name="notes.md", status="DONE", created_pages=["real", "hidden"], owner=owner))
    await db_session.commit()
    for slug in ["real", "hidden"]:
        await upsert_wiki_fts(db_session, slug, f"Title {slug}", "# Notes")
    await db_session.execute(text("UPDATE wiki_fts SET sensitive = true, owner = 'bob@bi.group' WHERE slug = 'hidden'"))
    await db_session.commit()
    session = await create_twin_session(db_session, case_id="case", persona_ids=[], created_by=owner)
    await append_twin_message(db_session, session_id=session.id, role="persona", persona_id=None, seq=0, content={"text": "Answer [[real]] [[hidden]]", "cite": "[Source Document]"})
    await append_twin_message(db_session, session_id=session.id, role="persona", persona_id=None, seq=1, content={"text": "Legacy [Source Document]"})
    async def database():
        yield db_session
    app.dependency_overrides[get_db] = database
    app.dependency_overrides[get_user_key] = lambda: owner
    try:
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as http:
            listing = await http.get("/api/v1/twin/sessions")
            assert listing.status_code == 200
            response = await http.get(f"/api/v1/twin/sessions/{session.id}/messages")
            assert response.status_code == 200
            content = response.json()[0]["content"]
            assert content["citations"] == [{"anchor": "real", "title": "Title real"}]
            assert "hidden" not in content["text"]
            assert response.json()[1]["content"]["citation_unavailable"] is True
            app.dependency_overrides[get_user_key] = lambda: "bob@bi.group"
            assert (await http.get(f"/api/v1/twin/sessions/{session.id}/messages")).status_code == 404
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_user_key, None)
