"""Chat/ask endpoints for the llm-wiki-frontend bridge (MVP)."""

from typing import Literal

from fastapi import Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.api.deps import get_db, get_user_key, get_user_title
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
        citation_quotes={c.anchor: c.quote for c in response.citations if c.quote},
        citation_cases={
            c.anchor: {"id": c.case_id, "title": c.case_title}
            for c in response.citations
            if c.case_id
        },
    )


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class AskBody(BaseModel):
    """Request body for document/card ask endpoints."""

    question: str = Field(min_length=1, max_length=2000)
    language: Literal["ru", "en", "kk"] = "ru"
    mode: Literal["library", "expert", "advisor"] = "expert"
    # Case ask only: restrict retrieval to this subset of the case's documents
    # (the panel's source checkboxes). Semantics matter (BUG-01):
    #   None → field not sent (older clients) → answer over every document;
    #   []   → the user unchecked EVERYTHING → refuse instead of silently
    #          answering over the whole case.
    doc_ids: list[str] | None = None


class Citation(BaseModel):
    """A citation anchor pointing to a wiki slug, with its human display title
    and a short supporting quote (for the [n] hover card + reader highlight)."""

    anchor: str
    title: str = ""
    quote: str | None = None
    # The case the cited source belongs to — powers the "source case" chip shown
    # next to the citation (click → open that case). None when it's not in a case.
    case_id: str | None = None
    case_title: str | None = None


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
    title: str = Depends(get_user_title),
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
    from llm_wiki.storage.metadata import case_for_file

    llm = LLMClient()
    try:
        store = EmbeddingStore(llm_client=llm)
        agent = AnswerAgent(llm, store)
        result = await agent.answer_for_document(
            question=body.question,
            document=fr,
            language=body.language,
            file_id=document_id,
            title=title,
        )
    finally:
        await llm.aclose()

    # The document belongs to at most one case — label every citation with it.
    owning_case = await case_for_file(db, document_id)
    citations = [Citation(anchor=s.slug, title=s.title, quote=s.quote) for s in result.sources]
    if owning_case:
        for c in citations:
            c.case_id, c.case_title = owning_case
    response = DocAskResponse(
        answer=result.answer,
        citations=citations,
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
    title: str = Depends(get_user_title),
) -> DocAskResponse:
    """Alias: card_id is treated as document_id."""
    return await ask_document(
        document_id=card_id, body=body, db=db, user_key=user_key, title=title
    )


@router.post("/cases/{case_id}/ask", response_model=DocAskResponse)
async def ask_case(
    case_id: str,
    body: AskBody,
    db: AsyncSession = Depends(get_db),
    user_key: str = Depends(get_user_key),
    title: str = Depends(get_user_title),
) -> DocAskResponse:
    """Ask a question scoped to a whole case (NotebookLM-style, across all docs)."""
    from llm_wiki.storage.metadata import CaseRecord

    case = await db.get(CaseRecord, case_id)
    if not case:
        raise HTTPException(404, "Case not found")

    doc_ids = list(case.doc_ids or [])
    # Honour the panel's source selection (BUG-01). The old guard here was
    # `if body.doc_ids:` — an explicit empty selection ([] is falsy) silently
    # fell through to the WHOLE case, which is exactly the UAT repro
    # («Контекст: 0 из 3» → полный ответ с цитатами). Distinguish:
    #   None      → field absent → all documents (backwards compatible);
    #   []        → everything unchecked → honest refusal, no LLM call;
    #   [ids]     → intersect with this case's documents; an intersection of
    #               zero (stale client state, foreign ids) refuses too rather
    #               than quietly widening back to the full case.
    if body.doc_ids is not None:
        selected = [d for d in body.doc_ids if d in set(doc_ids)]
        if not selected:
            refuse = {
                "ru": "Источники не выбраны. Отметьте хотя бы один материал в панели "
                      "слева — и я отвечу по нему.",
                "kk": "Дереккөздер таңдалмаған. Сол жақ панельден кемінде бір материалды "
                      "белгілеңіз — сол бойынша жауап беремін.",
                "en": "No sources are selected. Check at least one material in the left "
                      "panel and I will answer from it.",
            }
            response = DocAskResponse(
                answer=refuse.get(body.language, refuse["en"]),
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
        doc_ids = selected
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
            title=title,
            documents=documents,
            language=body.language,
            file_id=case_id,
        )
    finally:
        await llm.aclose()

    # A case answer only retrieves from this case's own documents, so every
    # citation belongs to this case — label them with it for the source chip.
    citations = [Citation(anchor=s.slug, title=s.title, quote=s.quote) for s in result.sources]
    for c in citations:
        c.case_id, c.case_title = case.id, case.title
    response = DocAskResponse(
        answer=result.answer,
        citations=citations,
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
