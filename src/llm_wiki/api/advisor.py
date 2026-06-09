"""Legacy-prefix advisor endpoint for AdvisorPanel.tsx.

AdvisorPanel.tsx calls ``POST /api/advisor/ask`` (not under ``/api/v1/``).
This router is registered in ``main.py`` with prefix ``/api``, making the
full path ``/api/advisor/ask``.

The endpoint streams tokens via SSE — same format as Phase 3 chat:
  data: {"token": "..."}
  data: {"done": true, "answer": "...", "sources": [...], "follow_ups": [...]}
  data: {"error": "..."}

Rate-limited to 10 requests per minute per email (same limiter as POST /search).
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncGenerator
from typing import Literal

import structlog
from fastapi import APIRouter, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from llm_wiki.config import settings
from llm_wiki.llm.chunk_store import ChunkHit, ChunkStore
from llm_wiki.llm.client import LLMClient
from llm_wiki.utils.rate_limit import advisor_limiter

logger = structlog.get_logger(__name__)
router = APIRouter()

_PROMPTS_DIR = LLMClient.PROMPTS_DIR

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


class AdvisorRequest(BaseModel):
    query: str
    language: Literal["ru", "en", "kk"] = "ru"


def _build_chunks_text(hits: list[ChunkHit]) -> str:
    return "\n".join(f"[chunk_{i}] {h.text}" for i, h in enumerate(hits))


def _parse_follow_ups(text: str) -> list[str]:
    m = re.search(r'\{"follow_ups"\s*:\s*(\[.*?\])\s*\}', text, re.DOTALL)
    if not m:
        return []
    try:
        return [str(q) for q in json.loads(m.group(1))]
    except (json.JSONDecodeError, ValueError):
        return []


def _strip_json_blocks(text: str) -> str:
    return re.sub(
        r'\s*\{["\s]*"follow_ups"\s*:\s*\[.*?\]\s*\}',
        "",
        text,
        flags=re.DOTALL,
    ).strip()


async def _advisor_generator(
    body: AdvisorRequest,
    email: str,
) -> AsyncGenerator[str, None]:
    # Rate-limit check — inside the generator so errors surface as SSE events
    user_key = email or "anonymous"
    allowed, retry = advisor_limiter.check(user_key)
    if not allowed:
        yield f'data: {json.dumps({"error": f"Rate limit exceeded. Retry in {retry}s"})}\n\n'
        return

    llm = LLMClient()
    full_text = ""
    hits: list[ChunkHit] = []

    try:
        chunk_store = ChunkStore(chroma_path=settings.chroma_dir, llm_client=llm)
        hits = chunk_store.query(text=body.query, top_k=15)

        if not hits:
            payload = json.dumps({
                "done": True,
                "insufficient_evidence": True,
                "answer": "Контекст не содержит информации для ответа на этот вопрос.",
                "contact": "knowledge-team@bi.group",
                "sources": [],
                "follow_ups": [],
            })
            yield f"data: {payload}\n\n"
            return

        system = "Ты — бизнес-советник по корпоративной базе знаний BI AQYL."
        prompt = (
            (_PROMPTS_DIR / "advisor.md")
            .read_text(encoding="utf-8")
            .format(
                query=body.query,
                chunks=_build_chunks_text(hits),
                language=body.language,
            )
        )

        async for token in llm.stream_completion(system=system, prompt=prompt):
            full_text += token
            yield f"data: {json.dumps({'token': token})}\n\n"

        follow_ups = _parse_follow_ups(full_text)
        clean_answer = _strip_json_blocks(full_text)

        sources = [
            {
                "title": h.title or h.slug,
                "content_type": "document",
                "path": None,
                "chunk_id": f"chunk_{i}",
            }
            for i, h in enumerate(hits)
        ]

        done = {
            "done": True,
            "answer": clean_answer,
            "sources": sources,
            "follow_ups": follow_ups,
        }
        yield f"data: {json.dumps(done)}\n\n"

    except Exception as exc:  # noqa: BLE001
        logger.error("advisor_ask_error", error=str(exc))
        yield f'data: {json.dumps({"error": str(exc)})}\n\n'
    finally:
        await llm.aclose()


@router.post("/advisor/ask", tags=["advisor"], response_model=None)
async def advisor_ask(
    body: AdvisorRequest,
    x_user_email: str = Header("", alias="X-User-Email"),
) -> StreamingResponse:
    """SSE endpoint for AdvisorPanel.tsx.

    Streams the advisor response token-by-token using the ``advisor.md`` prompt
    and top-15 chunk retrieval from ChromaDB.
    """
    return StreamingResponse(
        _advisor_generator(body, x_user_email),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )
