"""Case auto-tagger — classify a case against the fixed taxonomy (item B, part 2).

Given a case's title + the text of its materials, an LLM picks the relevant tags
from ``llm_wiki.taxonomy`` (never invents new ones). Used both by the per-case
Celery task (on create / doc-add) and the one-off backfill over existing cases.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import structlog

from llm_wiki.llm.client import LLMClient
from llm_wiki.taxonomy import CASE_TAGS, clean_tags

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from llm_wiki.storage.metadata import CaseRecord

logger = structlog.get_logger(__name__)

# Structured output — OpenAI enforces the schema so the payload is valid JSON.
_TAGS_SCHEMA = {
    "type": "object",
    "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
    "required": ["tags"],
    "additionalProperties": False,
}
_TAXONOMY_BLOCK = "\n".join(f"- {name}: {desc}" for name, desc in CASE_TAGS)

_MAX_DOCS = 8          # cap docs pulled per case
_MAX_PAGES_PER_DOC = 3  # cap wiki pages per doc
_MAX_PAGE_CHARS = 2000  # per page
_MAX_CONTENT_CHARS = 6000  # total content sent to the model


async def gather_case_text(case: "CaseRecord", session: "AsyncSession") -> str:
    """Assemble the classifier's input: the case title, its documents' names and
    their processed wiki bodies (when available). Best-effort — a doc that isn't
    processed yet simply contributes less."""
    from llm_wiki.storage import wiki_store
    from llm_wiki.storage.metadata import get_file_record

    parts: list[str] = []
    for doc_id in (case.doc_ids or [])[:_MAX_DOCS]:
        rec = await get_file_record(session, doc_id)
        if rec is None:
            continue
        parts.append(rec.original_name)
        for slug in (rec.created_pages or [])[:_MAX_PAGES_PER_DOC]:
            try:
                body = wiki_store.get_page(slug, caller=case.owner)
            except Exception:  # noqa: BLE001 — content is best-effort
                body = None
            if body:
                parts.append(body[:_MAX_PAGE_CHARS])
    return "\n\n".join(parts)[:_MAX_CONTENT_CHARS]


async def classify_case_tags(
    title: str, content: str, llm: LLMClient, *, file_id: str = "tagger"
) -> list[str]:
    """Return the relevant taxonomy tags for a case. Cleaned (unknown tags
    dropped). Never raises — returns [] on any failure, so auto-tagging can't
    break case creation or the backfill."""
    try:
        prompt = llm.load_prompt(
            "case_tags",
            taxonomy=_TAXONOMY_BLOCK,
            title=(title or "")[:500],
            content=(content or "").strip() or "(материалы ещё не обработаны)",
        )
        text, _usage = await llm.complete(
            prompt=prompt,
            system="Ты классификатор бизнес-кейсов. Возвращай только валидный JSON.",
            file_id=file_id,
            agent_type="tagger",
            response_format="json",
            json_schema=_TAGS_SCHEMA,
            schema_name="case_tags",
        )
        parsed = json.loads(text)
        return clean_tags(parsed.get("tags", []))
    except Exception as exc:  # noqa: BLE001
        logger.warning("autotag_classify_failed", file_id=file_id, error=str(exc))
        return []
