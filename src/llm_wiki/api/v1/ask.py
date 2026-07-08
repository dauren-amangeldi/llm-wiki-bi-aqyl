"""Chat/ask endpoints for the llm-wiki-frontend bridge (MVP)."""

from typing import Literal

from fastapi import Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.api.deps import get_db, get_user_key
from llm_wiki.api.v1 import router
from llm_wiki.storage.metadata import FileRecord, append_chat_message


async def _persist_turn(
    db: AsyncSession,
    *,
    user_key: str,
    scope_type: str,
    scope_id: str,
    question: str,
    response: "DocAskResponse",
) -> None:
    """Save the user question and assistant answer to chat history."""
    await append_chat_message(
        db,
        user_key=user_key,
        scope_type=scope_type,
        scope_id=scope_id,
        role="user",
        text_body=question,
    )
    await append_chat_message(
        db,
        user_key=user_key,
        scope_type=scope_type,
        scope_id=scope_id,
        role="assistant",
        text_body=response.answer,
        citations=[c.anchor for c in response.citations],
    )


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
    user_key: str = Depends(get_user_key),
    scope_type: str = "document",
) -> DocAskResponse:
    """Ask a question scoped to a specific document using its wiki pages."""
    fr = await db.get(FileRecord, document_id)
    if not fr:
        raise HTTPException(404, "Document not found")

    if fr.status not in ("DONE", "LOGGED", "WRITTEN"):
        template = _PROCESSING_MSG.get(body.language, _PROCESSING_MSG["en"])
        response = DocAskResponse(
            answer=template.format(status=fr.status),
            citations=[],
            follow_ups=[],
            insufficient_evidence=False,
            contact=None,
        )
        await _persist_turn(
            db,
            user_key=user_key,
            scope_type=scope_type,
            scope_id=document_id,
            question=body.question,
            response=response,
        )
        return response

    from llm_wiki.agents.answer import AnswerAgent
    from llm_wiki.config import settings
    from llm_wiki.llm.client import LLMClient
    from llm_wiki.llm.embeddings import EmbeddingStore

    llm = LLMClient()
    try:
        store = EmbeddingStore(llm_client=llm)
        agent = AnswerAgent(llm, store)
        result = await agent.answer_for_document(
            question=body.question,
            document=fr,
            language=body.language,
            file_id=document_id,
        )
    finally:
        await llm.aclose()

    response = DocAskResponse(
        answer=result.answer,
        citations=[Citation(anchor=s.slug) for s in result.sources],
        follow_ups=[],
        insufficient_evidence=False,
        contact=None,
    )
    await _persist_turn(
        db,
        user_key=user_key,
        scope_type=scope_type,
        scope_id=document_id,
        question=body.question,
        response=response,
    )
    return response


@router.post("/cards/{card_id}/ask", response_model=DocAskResponse)
async def ask_card(
    card_id: str,
    body: AskBody,
    db: AsyncSession = Depends(get_db),
    user_key: str = Depends(get_user_key),
) -> DocAskResponse:
    """Alias: card_id is treated as document_id."""
    return await ask_document(
        document_id=card_id, body=body, db=db, user_key=user_key
    )


@router.post("/cases/{case_id}/ask", response_model=DocAskResponse)
async def ask_case(
    case_id: str,
    body: AskBody,
    db: AsyncSession = Depends(get_db),
    user_key: str = Depends(get_user_key),
) -> DocAskResponse:
    """Ask a question scoped to a whole case (NotebookLM-style, across all docs)."""
    from llm_wiki.storage.metadata import CaseRecord

    case = await db.get(CaseRecord, case_id)
    if not case:
        raise HTTPException(404, "Case not found")

    doc_ids = list(case.doc_ids or [])
    documents: list[FileRecord] = []
    for did in doc_ids:
        fr = await db.get(FileRecord, did)
        if fr is not None:
            documents.append(fr)

    if not documents:
        empty = {
            "ru": "В этом кейсе пока нет обработанных материалов. Загрузите файл и дождитесь обработки.",
            "kk": "Бұл кейсте әзірге өңделген материалдар жоқ. Файл жүктеп, өңделуін күтіңіз.",
            "en": "This case has no processed materials yet. Upload a file and wait for processing.",
        }
        response = DocAskResponse(
            answer=empty.get(body.language, empty["en"]),
            citations=[],
            follow_ups=[],
            insufficient_evidence=True,
            contact=None,
        )
        await _persist_turn(
            db,
            user_key=user_key,
            scope_type="case",
            scope_id=case_id,
            question=body.question,
            response=response,
        )
        return response

    from llm_wiki.agents.answer import AnswerAgent
    from llm_wiki.config import settings
    from llm_wiki.llm.client import LLMClient
    from llm_wiki.llm.embeddings import EmbeddingStore

    llm = LLMClient()
    try:
        store = EmbeddingStore(llm_client=llm)
        agent = AnswerAgent(llm, store)
        result = await agent.answer_for_case(
            question=body.question,
            documents=documents,
            language=body.language,
            file_id=case_id,
        )
    finally:
        await llm.aclose()

    response = DocAskResponse(
        answer=result.answer,
        citations=[Citation(anchor=s.slug) for s in result.sources],
        follow_ups=[],
        insufficient_evidence=not result.sources,
        contact=None,
    )
    await _persist_turn(
        db,
        user_key=user_key,
        scope_type="case",
        scope_id=case_id,
        question=body.question,
        response=response,
    )
    return response
