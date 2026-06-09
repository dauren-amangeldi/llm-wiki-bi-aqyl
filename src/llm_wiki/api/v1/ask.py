"""Chat/ask endpoints for the llm-wiki-frontend bridge (MVP)."""

from typing import Literal

from fastapi import Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.api.deps import get_db
from llm_wiki.api.v1 import router
from llm_wiki.storage.metadata import FileRecord


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class AskBody(BaseModel):
    """Request body for document/card ask endpoints."""

    question: str = Field(min_length=1, max_length=2000)
    language: Literal["ru", "en", "kk"] = "ru"
    mode: Literal["library", "expert", "advisor"] = "expert"


class Citation(BaseModel):
    """A citation anchor pointing to a wiki slug."""

    anchor: str


class DocAskResponse(BaseModel):
    """Response schema for the document ask endpoints."""

    answer: str
    citations: list[Citation] = []
    follow_ups: list[str] = []
    insufficient_evidence: bool = False
    contact: str | None = None


_PROCESSING_MSG = {
    "ru": "Документ ещё обрабатывается (статус: {status}). Попробуйте через минуту.",
    "kk": "Құжат әлі өңделуде (статус: {status}). Бір минуттан кейін көріңіз.",
    "en": "Document is still being processed (status: {status}). Try again in a minute.",
}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/documents/{document_id}/ask", response_model=DocAskResponse)
async def ask_document(
    document_id: str,
    body: AskBody,
    db: AsyncSession = Depends(get_db),
) -> DocAskResponse:
    """Ask a question scoped to a specific document using its wiki pages."""
    fr = await db.get(FileRecord, document_id)
    if not fr:
        raise HTTPException(404, "Document not found")

    if fr.status not in ("DONE", "LOGGED", "WRITTEN"):
        template = _PROCESSING_MSG.get(body.language, _PROCESSING_MSG["en"])
        return DocAskResponse(
            answer=template.format(status=fr.status),
            citations=[],
            follow_ups=[],
            insufficient_evidence=False,
            contact=None,
        )

    from llm_wiki.agents.answer import AnswerAgent
    from llm_wiki.config import settings
    from llm_wiki.llm.client import LLMClient
    from llm_wiki.llm.embeddings import EmbeddingStore

    llm = LLMClient()
    try:
        store = EmbeddingStore(chroma_path=settings.chroma_dir, llm_client=llm)
        agent = AnswerAgent(llm, store)
        result = await agent.answer_for_document(
            question=body.question,
            document=fr,
            language=body.language,
            file_id=document_id,
        )
    finally:
        await llm.aclose()

    return DocAskResponse(
        answer=result.answer,
        citations=[Citation(anchor=s.slug) for s in result.sources],
        follow_ups=[],
        insufficient_evidence=False,
        contact=None,
    )


@router.post("/cards/{card_id}/ask", response_model=DocAskResponse)
async def ask_card(
    card_id: str,
    body: AskBody,
    db: AsyncSession = Depends(get_db),
) -> DocAskResponse:
    """Alias: card_id is treated as document_id."""
    return await ask_document(document_id=card_id, body=body, db=db)
