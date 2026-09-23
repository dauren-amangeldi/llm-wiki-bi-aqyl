"""Automatic naming must never overwrite a concurrent manual edit."""
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException
from llm_wiki.api.v1.cases import generate_case_title


class CaseTitleTests(unittest.IsolatedAsyncioTestCase):
    async def run_request(self, *, owner="alice", docs=None, raw='{"title":"  Партнёрство по AI-продукту  "}', rowcount=1):
        row = SimpleNamespace(id="case-1", title="report · second", owner=owner,
                              doc_ids=["f1", "f2"] if docs is None else docs)
        names = MagicMock()
        names.scalars.return_value.all.return_value = ["report.pdf", "second.docx"]
        db = SimpleNamespace(get=AsyncMock(return_value=row),
                             execute=AsyncMock(side_effect=[names, SimpleNamespace(rowcount=rowcount)]),
                             commit=AsyncMock())
        llm = SimpleNamespace(complete=AsyncMock(return_value=(raw, {})), aclose=AsyncMock())
        with patch("llm_wiki.agents.tagger.gather_case_text", AsyncMock(return_value="Материалы о партнёрстве")), \
             patch("llm_wiki.llm.client.LLMClient", return_value=llm):
            result = await generate_case_title("case-1", db, "alice")
        return result, db, llm

    async def test_names_from_source_set(self):
        result, db, llm = await self.run_request()
        self.assertEqual(result["title"], "Партнёрство по AI-продукту")
        prompt = llm.complete.call_args.kwargs["prompt"]
        self.assertIn("report.pdf", prompt)
        self.assertIn("second.docx", prompt)
        self.assertIn("Материалы о партнёрстве", prompt)
        db.commit.assert_awaited_once()
        llm.aclose.assert_awaited_once()

    async def test_foreign_case_is_forbidden(self):
        with self.assertRaises(HTTPException) as error:
            await self.run_request(owner="bob")
        self.assertEqual(error.exception.status_code, 403)

    async def test_empty_case_is_rejected(self):
        with self.assertRaises(HTTPException) as error:
            await self.run_request(docs=[])
        self.assertEqual(error.exception.status_code, 422)

    async def test_invalid_model_output_is_recoverable(self):
        with self.assertRaises(HTTPException) as error:
            await self.run_request(raw='{"title": ""}')
        self.assertEqual(error.exception.status_code, 502)

    async def test_cleans_file_style_model_output_before_saving(self):
        result, db, _ = await self.run_request(
            raw='{"title": "02-25_Modern_Marketing_Strategy.pdf"}'
        )
        self.assertEqual(result["title"], "Modern Marketing Strategy")
        statement = db.execute.call_args_list[-1].args[0]
        self.assertEqual(statement.compile().params["title"], result["title"])

    async def test_rejects_output_that_is_empty_after_cleaning(self):
        with self.assertRaises(HTTPException) as error:
            await self.run_request(raw='{"title": "02-25___"}')
        self.assertEqual(error.exception.status_code, 502)

    async def test_concurrent_edit_is_not_overwritten(self):
        with self.assertRaises(HTTPException) as error:
            await self.run_request(rowcount=0)
        self.assertEqual(error.exception.status_code, 409)


if __name__ == "__main__":
    unittest.main()
