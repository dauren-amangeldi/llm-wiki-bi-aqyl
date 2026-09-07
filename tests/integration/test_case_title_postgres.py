"""Run against an isolated PostgreSQL via CASE_TITLE_TEST_DATABASE_URL."""
import json
import os
import unittest
from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from sqlalchemy import delete, update
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from llm_wiki.api.v1.cases import generate_case_title
from llm_wiki.storage.metadata import Base, CaseRecord, FileRecord


@unittest.skipUnless(os.getenv("CASE_TITLE_TEST_DATABASE_URL"), "isolated PostgreSQL URL required")
class CaseTitlePostgresTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = create_async_engine(os.environ["CASE_TITLE_TEST_DATABASE_URL"])
        async with self.engine.begin() as conn:
            await conn.run_sync(lambda c: Base.metadata.create_all(c, tables=[CaseRecord.__table__, FileRecord.__table__]))
        self.case_id = f"title-test-{uuid4()}"
        self.db = AsyncSession(self.engine, expire_on_commit=False)
        self.db.add(CaseRecord(id=self.case_id, title="Original", doc_ids=["source-1"], owner="title-test"))
        await self.db.commit()

    async def asyncTearDown(self):
        await self.db.close()
        async with self.engine.begin() as conn:
            await conn.execute(delete(CaseRecord).where(CaseRecord.id == self.case_id))
        await self.engine.dispose()

    async def generate(self, concurrent_change=None):
        async def complete(**kwargs):
            if concurrent_change:
                async with self.engine.begin() as conn:
                    await conn.execute(update(CaseRecord).where(CaseRecord.id == self.case_id).values(**concurrent_change))
            return json.dumps({"title": "Generated title"}), {}

        llm = SimpleNamespace(complete=complete, aclose=AsyncMock())
        with patch("llm_wiki.agents.tagger.gather_case_text", AsyncMock(return_value="Source text")), \
             patch("llm_wiki.llm.client.LLMClient", return_value=llm):
            return await generate_case_title(self.case_id, self.db, "title-test")

    async def test_updates_title_with_json_source_column(self):
        result = await self.generate()
        self.assertEqual(result["title"], "Generated title")
        self.db.expire_all()
        self.assertEqual((await self.db.get(CaseRecord, self.case_id)).title, "Generated title")

    async def test_preserves_concurrent_manual_title(self):
        with self.assertRaises(HTTPException) as error:
            await self.generate({"title": "Manual title"})
        self.assertEqual(error.exception.status_code, 409)
        self.db.expire_all()
        self.assertEqual((await self.db.get(CaseRecord, self.case_id)).title, "Manual title")

    async def test_rejects_changed_source_set(self):
        with self.assertRaises(HTTPException) as error:
            await self.generate({"doc_ids": ["source-2"]})
        self.assertEqual(error.exception.status_code, 409)
        self.db.expire_all()
        row = await self.db.get(CaseRecord, self.case_id)
        self.assertEqual(row.title, "Original")
        self.assertEqual(row.doc_ids, ["source-2"])
