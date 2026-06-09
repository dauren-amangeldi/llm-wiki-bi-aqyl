"""Chat/ask endpoints for the llm-wiki-frontend bridge (MVP)."""

from typing import Literal

from fastapi import Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.api.deps import get_db
from llm_wiki.api.v1 import router
from llm_wiki.storage.metadata import FileRecord


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class AskBody(BaseModel):
    """Request body for document/card ask endpoints."""

    question: str
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _ask_via_answer_agent(question: str) -> DocAskResponse:
    """Invoke the existing AnswerAgent and convert its result for the frontend."""
    from llm_wiki.agents.answer import AnswerAgent
    from llm_wiki.config import settings
    from llm_wiki.llm.chunk_store import ChunkStore
    from llm_wiki.llm.client import LLMClient
    from llm_wiki.llm.embeddings import EmbeddingStore

    llm = LLMClient()
    try:
        embedding_store = EmbeddingStore(chroma_path=settings.chroma_dir, llm_client=llm)
        chunk_store = ChunkStore(chroma_path=settings.chroma_dir, llm_client=llm)
        agent = AnswerAgent(llm, embedding_store, chunk_store=chunk_store)
        result = await agent.answer(question=question, top_k=5)
    finally:
        await llm.aclose()

    insufficient = result.confidence == "low" and not result.sources
    return DocAskResponse(
        answer=result.answer,
        citations=[Citation(anchor=s.slug) for s in result.sources],
        follow_ups=[],
        insufficient_evidence=insufficient,
        contact="knowledge-team@bi.group" if insufficient else None,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/documents/{document_id}/ask", response_model=DocAskResponse)
async def ask_document(
    document_id: str,
    body: AskBody,
    db: AsyncSession = Depends(get_db),
) -> DocAskResponse:
    """Ask a question scoped to a specific document (RAG over full wiki for MVP)."""
    fr = await db.get(FileRecord, document_id)
    if not fr:
        raise HTTPException(404, "Document not found")
    return await _ask_via_answer_agent(body.question)


@router.post("/cards/{card_id}/ask", response_model=DocAskResponse)
async def ask_card(
    card_id: str,
    body: AskBody,
    db: AsyncSession = Depends(get_db),
) -> DocAskResponse:
    """Alias: card_id is treated as document_id."""
    return await ask_document(document_id=card_id, body=body, db=db)
