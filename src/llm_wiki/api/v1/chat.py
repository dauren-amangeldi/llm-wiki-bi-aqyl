"""Phase 3 — Document chat endpoints.

POST /documents/{document_id}/ask
POST /cards/{card_id}/ask   (alias)

Both endpoints support two response modes selected by the query parameter
``?stream=true``:

* Default (no ``?stream=true``): standard JSON response — works with the
  current ``apiFetch`` calls in ``ChatColumn.tsx``.
* ``?stream=true``: Server-Sent Events stream — works with ``useSSEStream.ts``.
  Token events arrive incrementally; a final ``done`` event carries the full
  answer, sources, citations, and follow-ups.

Retrieval strategy
------------------
1. Look up the ``FileRecord`` to get the list of wiki slugs produced by this
   document's ingestion pipeline (``created_pages`` + ``updated_pages``).
2. Query the ``chunks`` ChromaDB collection with a slug filter so only chunks
   from this document's wiki pages are searched.
3. If no chunks are found (document not yet indexed), return
   ``insufficient_evidence=True`` without calling the LLM.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncGenerator
from typing import Any, Literal

import structlog
from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.api.deps import get_db
from llm_wiki.config import settings
from llm_wiki.llm.chunk_store import ChunkHit, ChunkStore
from llm_wiki.llm.client import LLMClient
from llm_wiki.storage.metadata import get_file_record

# Resolve prompts directory at import time so it survives LLMClient mocking in tests.
_PROMPTS_DIR = LLMClient.PROMPTS_DIR

logger = structlog.get_logger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_INSUFFICIENT_CONTACT = "knowledge-team@bi.group"

_INSUFFICIENT_ANSWER: dict[str, str] = {
    "ru": "Контекст не содержит информации для ответа на этот вопрос. Обратитесь к команде знаний.",
    "en": "The context does not contain information to answer this question. Contact the knowledge team.",
    "kk": "Мазмұн бұл сұраққа жауап беруге жеткіліксіз. Білім тобына хабарласыңыз.",
}

_MODE_DESC: dict[str, dict[str, str]] = {
    "library": {
        "ru": "library — кратко по фактам, без лишних пояснений",
        "en": "library — brief and factual, no extra commentary",
        "kk": "library — фактілерге негізделген қысқаша жауап",
    },
    "expert": {
        "ru": "expert — подробный разбор с обоснованием и деталями",
        "en": "expert — detailed analysis with reasoning and details",
        "kk": "expert — толық талдау мен дәлелдемемен",
    },
    "advisor": {
        "ru": "advisor — практические рекомендации и конкретные следующие шаги",
        "en": "advisor — practical recommendations and concrete next steps",
        "kk": "advisor — практикалық ұсыныстар мен нақты келесі қадамдар",
    },
}

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}

# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    question: str
    language: Literal["ru", "en", "kk"] = "ru"
    mode: Literal["library", "expert", "advisor"] = "expert"


# ---------------------------------------------------------------------------
# Text-parsing helpers
# ---------------------------------------------------------------------------


def _build_chunks_text(hits: list[ChunkHit]) -> str:
    """Format chunk hits into the prompt's {chunks} block."""
    parts: list[str] = []
    for i, hit in enumerate(hits):
        header = f"[chunk_{i}]"
        if hit.section:
            header += f" ({hit.section})"
        parts.append(f"{header}\n{hit.text}")
    return "\n\n---\n\n".join(parts)


def _parse_follow_ups(text: str) -> list[str]:
    """Extract follow_ups list from any embedded JSON block in text."""
    match = re.search(r'\{"follow_ups"\s*:\s*(\[.*?\])\s*\}', text, re.DOTALL)
    if not match:
        return []
    try:
        result = json.loads(match.group(1))
        return [str(q) for q in result] if isinstance(result, list) else []
    except (json.JSONDecodeError, ValueError):
        return []


def _parse_citations(text: str) -> list[dict[str, str]]:
    """Return unique [chunk_N] references from text as citation anchor dicts."""
    seen: set[str] = set()
    citations: list[dict[str, str]] = []
    for m in re.finditer(r'\[chunk_\d+\]', text):
        anchor = m.group(0)[1:-1]  # strip outer brackets
        if anchor not in seen:
            seen.add(anchor)
            citations.append({"anchor": anchor})
    return citations


def _check_insufficient(text: str) -> bool:
    """Return True if the LLM response signals insufficient evidence."""
    return '"insufficient_evidence"' in text and "true" in text.lower()


def _strip_json_blocks(text: str) -> str:
    """Remove embedded JSON follow_ups block from the final answer text."""
    return re.sub(
        r'\s*\{["\s]*"follow_ups"\s*:\s*\[.*?\]\s*\}',
        "",
        text,
        flags=re.DOTALL,
    ).strip()


# ---------------------------------------------------------------------------
# Core retrieval (shared between streaming and non-streaming paths)
# ---------------------------------------------------------------------------


async def _get_wiki_slugs(document_id: str, session: AsyncSession) -> list[str] | None:
    """Return the wiki slugs produced by this document's ingestion.

    Returns:
        List of slugs (may be empty if ingestion not yet complete).
        None if the document record is missing or deleted.
    """
    fr = await get_file_record(session, document_id)
    if fr is None or fr.status == "deleted":
        return None
    return list(set((fr.created_pages or []) + (fr.updated_pages or [])))


def _build_prompt(hits: list[ChunkHit], body: AskRequest) -> tuple[str, str]:
    """Return (system, user_prompt) for the LLM call."""
    system = "Ты — экспертный ассистент корпоративной базы знаний BI AQYL."
    lang = body.language
    mode_desc = _MODE_DESC.get(body.mode, _MODE_DESC["expert"]).get(lang, body.mode)
    chunks_text = _build_chunks_text(hits)
    user_prompt = (
        (_PROMPTS_DIR / "chat_document.md")
        .read_text(encoding="utf-8")
        .format(
            mode_desc=mode_desc,
            language=lang,
            chunks=chunks_text,
            question=body.question,
        )
    )
    return system, user_prompt


def _sources_payload(hits: list[ChunkHit], document_id: str) -> list[dict[str, Any]]:
    """Serialise hits into the ``sources`` list for the done event / JSON response."""
    return [
        {
            "chunk_id": f"chunk_{i}",
            "text": hit.text[:300],  # truncate for payload size
            "document_id": document_id,
            "slug": hit.slug,
            "section": hit.section,
        }
        for i, hit in enumerate(hits)
    ]


# ---------------------------------------------------------------------------
# SSE generator
# ---------------------------------------------------------------------------


async def _sse_insufficient(language: str) -> AsyncGenerator[str, None]:
    """Yield a single SSE done event for the insufficient-evidence case."""
    payload = json.dumps(
        {
            "done": True,
            "insufficient_evidence": True,
            "answer": _INSUFFICIENT_ANSWER.get(language, _INSUFFICIENT_ANSWER["ru"]),
            "contact": _INSUFFICIENT_CONTACT,
            "sources": [],
            "citations": [],
            "follow_ups": [],
        }
    )
    yield f"data: {payload}\n\n"


async def _sse_stream(
    llm: LLMClient,
    system: str,
    prompt: str,
    hits: list[ChunkHit],
    document_id: str,
    language: str,
) -> AsyncGenerator[str, None]:
    """Stream tokens then emit a final done event."""
    full_text = ""
    try:
        async for token in llm.stream_completion(system=system, prompt=prompt):
            full_text += token
            yield f"data: {json.dumps({'token': token})}\n\n"

        # Parse structured fields from the buffered response
        follow_ups = _parse_follow_ups(full_text)
        citations = _parse_citations(full_text)
        clean_answer = _strip_json_blocks(full_text)
        insufficient = _check_insufficient(full_text)

        done_payload: dict[str, Any] = {
            "done": True,
            "answer": clean_answer,
            "sources": _sources_payload(hits, document_id),
            "citations": citations,
            "follow_ups": follow_ups,
        }
        if insufficient:
            done_payload["insufficient_evidence"] = True
            done_payload["contact"] = _INSUFFICIENT_CONTACT

        yield f"data: {json.dumps(done_payload)}\n\n"

    except Exception as exc:  # noqa: BLE001
        logger.error("chat_stream_error", document_id=document_id, error=str(exc))
        yield f"data: {json.dumps({'error': str(exc)})}\n\n"
    finally:
        await llm.aclose()


# ---------------------------------------------------------------------------
# Endpoint logic (shared)
# ---------------------------------------------------------------------------


async def _handle_ask(
    document_id: str,
    body: AskRequest,
    stream: bool,
    session: AsyncSession,
) -> StreamingResponse | JSONResponse:
    """Core logic for both /documents/{id}/ask and /cards/{id}/ask."""
    # 1. Resolve document → wiki slugs
    wiki_slugs = await _get_wiki_slugs(document_id, session)

    if wiki_slugs is None:
        if stream:
            return StreamingResponse(
                _sse_insufficient(body.language),
                media_type="text/event-stream",
                headers=_SSE_HEADERS,
            )
        return JSONResponse(
            {
                "answer": _INSUFFICIENT_ANSWER.get(body.language, _INSUFFICIENT_ANSWER["ru"]),
                "insufficient_evidence": True,
                "contact": _INSUFFICIENT_CONTACT,
                "citations": [],
                "follow_ups": [],
            }
        )

    # 2. Query chunks (scoped to this document's wiki pages)
    llm = LLMClient()
    hits: list[ChunkHit] = []
    try:
        chunk_store = ChunkStore(chroma_path=settings.chroma_dir, llm_client=llm)
        if wiki_slugs:
            hits = chunk_store.query(
                text=body.question,
                top_k=5,
                file_id=document_id,
                slug_filter=wiki_slugs,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("chat_retrieval_error", document_id=document_id, error=str(exc))
        if not stream:
            await llm.aclose()

    # 3. Insufficient evidence: no chunks found
    if not hits:
        if stream:
            # aclose() is handled inside _sse_insufficient (it's a simple generator)
            await llm.aclose()
            return StreamingResponse(
                _sse_insufficient(body.language),
                media_type="text/event-stream",
                headers=_SSE_HEADERS,
            )
        await llm.aclose()
        return JSONResponse(
            {
                "answer": _INSUFFICIENT_ANSWER.get(body.language, _INSUFFICIENT_ANSWER["ru"]),
                "insufficient_evidence": True,
                "contact": _INSUFFICIENT_CONTACT,
                "citations": [],
                "follow_ups": [],
            }
        )

    # 4. Build prompt
    system, prompt = _build_prompt(hits, body)

    # 5a. Streaming path — return SSE; aclose() handled inside generator
    if stream:
        return StreamingResponse(
            _sse_stream(llm, system, prompt, hits, document_id, body.language),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )

    # 5b. Non-streaming path — call LLM synchronously, return JSON
    try:
        text, _usage = await llm.complete(
            prompt=prompt,
            system=system,
            file_id=document_id,
            agent_type="answer",
        )
        follow_ups = _parse_follow_ups(text)
        citations = _parse_citations(text)
        clean_answer = _strip_json_blocks(text)
        insufficient = _check_insufficient(text)

        payload: dict[str, Any] = {
            "answer": clean_answer,
            "sources": _sources_payload(hits, document_id),
            "citations": citations,
            "follow_ups": follow_ups,
        }
        if insufficient:
            payload["insufficient_evidence"] = True
            payload["contact"] = _INSUFFICIENT_CONTACT

        return JSONResponse(payload)
    finally:
        await llm.aclose()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/documents/{document_id}/ask", tags=["chat"], response_model=None)
async def ask_document(
    document_id: str,
    body: AskRequest,
    stream: bool = Query(False, alias="stream"),
    session: AsyncSession = Depends(get_db),
) -> StreamingResponse | JSONResponse:
    """Ask a question about a specific document using RAG over its wiki chunks.

    Response mode:
    - Default (no ``?stream=true``): standard JSON for ``ChatColumn.tsx`` / ``apiFetch``.
    - ``?stream=true``: SSE token stream for ``useSSEStream.ts``.
    """
    return await _handle_ask(document_id, body, stream, session)


@router.post("/cards/{card_id}/ask", tags=["chat"], response_model=None)
async def ask_card(
    card_id: str,
    body: AskRequest,
    stream: bool = Query(False, alias="stream"),
    session: AsyncSession = Depends(get_db),
) -> StreamingResponse | JSONResponse:
    """Alias for ``/documents/{document_id}/ask`` used when the material has a card_id."""
    return await _handle_ask(card_id, body, stream, session)
