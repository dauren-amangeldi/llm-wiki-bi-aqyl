"""Phase 4 — Global search endpoints.

GET  /search  — JSON response for SearchPanel / SearchBar (apiFetch).
               Returns SearchResult[] compatible with both SearchResultRaw
               and Material shapes used by the two frontend components.

POST /search  — SSE response for expert/advisor streaming modes.
               Always returns text/event-stream regardless of mode.

Retrieval strategy (both endpoints):
  library / GET: semantic search via EmbeddingStore (heading-level), top-10
                 unique document_ids, graceful fallback to SQL substring match.
  expert (POST):  same retrieval top-5 + LLM synthesis of wiki summaries.
  advisor (POST): top-15 chunks from ChunkStore + LLM advisor analysis
                  (rate-limited per email).
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, Literal

import structlog
from fastapi import APIRouter, Depends, Header, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.api.deps import get_db
from llm_wiki.config import settings
from llm_wiki.llm.chunk_store import ChunkHit, ChunkStore
from llm_wiki.llm.client import LLMClient
from llm_wiki.llm.embeddings import EmbeddingStore
from llm_wiki.storage.metadata import FileRecord
from llm_wiki.utils.rate_limit import advisor_limiter
from llm_wiki.utils.slugify import to_slug

logger = structlog.get_logger(__name__)
router = APIRouter()

_PROMPTS_DIR = LLMClient.PROMPTS_DIR

_EXT_TO_CT: dict[str, str] = {
    ".pdf": "pdf",
    ".md": "markdown",
}

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class SearchResult(BaseModel):
    """Compatible with both SearchResultRaw (SearchPanel) and Material (SearchBar)."""

    document_id: str
    document_title: str
    title: str
    snippet: str = ""
    scope: str = "internal"
    classification: str = ""
    score: float = 1.0
    content_type: str = "document"
    topic_id: str | None = None
    tags: list[str] = []
    # Material-compatible fields
    business_unit: str = "HQ"
    status: str = "DONE"
    created_at: str = ""
    updated_at: str | None = None
    source_language: str = "ru"
    topic_ids: list[str] = []
    title_i18n: dict[str, str] = {}
    author: str | None = None
    language: str = "ru"


class SearchRequest(BaseModel):
    query: str
    scope: Literal["all", "internal", "external"] = "all"
    tags: list[str] = []
    mode: Literal["library", "expert", "advisor"] = "library"
    language: Literal["ru", "en", "kk"] = "ru"


# ---------------------------------------------------------------------------
# Shared retrieval helpers
# ---------------------------------------------------------------------------


def _fr_to_result(fr: FileRecord, score: float, snippet: str = "") -> SearchResult:
    ext = Path(fr.original_name).suffix.lower()
    title = Path(fr.original_name).stem
    return SearchResult(
        document_id=fr.file_id,
        document_title=title,
        title=title,
        snippet=snippet,
        scope="internal",
        classification="",
        score=round(score, 4),
        content_type=_EXT_TO_CT.get(ext, "document"),
        business_unit="HQ",
        status=fr.status,
        created_at=fr.created_at.isoformat(),
        updated_at=fr.updated_at.isoformat() if fr.updated_at else None,
        title_i18n={"ru": title},
        language="ru",
    )


def _wiki_snippet(fr: FileRecord, max_chars: int = 200) -> str:
    """Return the first *max_chars* characters of the document's wiki page."""
    title = Path(fr.original_name).stem
    slug = to_slug(title)
    path = settings.wiki_dir / f"{slug}.md"
    if path.exists():
        return path.read_text(encoding="utf-8")[:max_chars]
    return ""


async def _semantic_search(
    q: str,
    session: AsyncSession,
    top_k: int = 10,
) -> list[tuple[FileRecord, float]]:
    """Embed *q* and return matching FileRecords with cosine similarity scores.

    Falls back to SQL LIKE search if the embedding collection is empty or
    the OpenAI API call fails.
    """
    slug_scores: dict[str, float] = {}

    try:
        llm = LLMClient()
        try:
            emb_store = EmbeddingStore(chroma_path=settings.chroma_dir, llm_client=llm)
            hits = emb_store.query(q.strip(), top_k=top_k)
            for hit in hits:
                slug_scores[hit.slug] = max(slug_scores.get(hit.slug, 0.0), hit.similarity)
        finally:
            await llm.aclose()
    except Exception as exc:  # noqa: BLE001
        logger.warning("search_semantic_failed", query=q[:80], error=str(exc))

    result = await session.execute(
        select(FileRecord)
        .where(FileRecord.status != "deleted")
        .order_by(FileRecord.created_at.desc())
        .limit(500)
    )
    all_records: list[FileRecord] = list(result.scalars().all())

    matched: list[tuple[FileRecord, float]] = []

    if slug_scores:
        for fr in all_records:
            fr_slugs = set((fr.created_pages or []) + (fr.updated_pages or []))
            common = fr_slugs & set(slug_scores.keys())
            if common:
                best = max(slug_scores[s] for s in common)
                matched.append((fr, best))
        matched.sort(key=lambda x: -x[1])
    else:
        q_lower = q.lower()
        for fr in all_records:
            if q_lower in fr.original_name.lower():
                matched.append((fr, 1.0))

    return matched[:top_k]


# ---------------------------------------------------------------------------
# SSE helpers (reuse pattern from Phase 3)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# GET /search  — JSON, used by SearchPanel + SearchBar
# ---------------------------------------------------------------------------


@router.get("/search", tags=["search"])
async def get_search(
    q: str = Query(""),
    scope: str = Query("all"),
    language: str = Query("ru"),
    session: AsyncSession = Depends(get_db),
) -> list[SearchResult]:
    """Semantic search returning JSON list.

    Satisfies both ``SearchResultRaw[]`` (SearchPanel) and ``Material[]``
    (SearchBar) consumers by including all required fields in one response.
    """
    if not q.strip():
        return []

    matched = await _semantic_search(q, session, top_k=10)

    results: list[SearchResult] = []
    for fr, score in matched:
        snippet = _wiki_snippet(fr)
        results.append(_fr_to_result(fr, score, snippet))

    return results


# ---------------------------------------------------------------------------
# POST /search  — SSE, modes: library / expert / advisor
# ---------------------------------------------------------------------------


@router.post("/search", tags=["search"], response_model=None)
async def post_search(
    body: SearchRequest,
    x_user_email: str = Header("", alias="X-User-Email"),
    session: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """SSE search endpoint — always returns text/event-stream.

    library: single done event with results list.
    expert:  token stream + done event with synthesised answer.
    advisor: rate-limited, top-15 chunk retrieval + advisor LLM analysis.
    """
    return StreamingResponse(
        _search_generator(body, x_user_email, session),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


async def _search_generator(
    body: SearchRequest,
    email: str,
    session: AsyncSession,
) -> AsyncGenerator[str, None]:
    """Drive the SSE stream for all three search modes."""
    try:
        if body.mode == "library":
            yield await _library_event(body, session)

        elif body.mode == "expert":
            async for event in _expert_stream(body, session):
                yield event

        elif body.mode == "advisor":
            # Rate-limit check
            user_key = email or "anonymous"
            allowed, retry = advisor_limiter.check(user_key)
            if not allowed:
                yield f'data: {json.dumps({"error": f"Rate limit. Retry in {retry}s"})}\n\n'
                return
            async for event in _advisor_stream(body, session):
                yield event

    except Exception as exc:  # noqa: BLE001
        logger.error("search_sse_error", mode=body.mode, error=str(exc))
        yield f'data: {json.dumps({"error": str(exc)})}\n\n'


# ── library ──────────────────────────────────────────────────────────────────


async def _library_event(body: SearchRequest, session: AsyncSession) -> str:
    matched = await _semantic_search(body.query, session, top_k=10)
    results = [_fr_to_result(fr, score, _wiki_snippet(fr)) for fr, score in matched]
    payload = {"done": True, "results": [r.model_dump() for r in results]}
    return f"data: {json.dumps(payload)}\n\n"


# ── expert ────────────────────────────────────────────────────────────────────


async def _expert_stream(
    body: SearchRequest, session: AsyncSession
) -> AsyncGenerator[str, None]:
    matched = await _semantic_search(body.query, session, top_k=5)

    summaries_parts: list[str] = []
    for fr, _ in matched:
        doc_id = fr.file_id
        title = Path(fr.original_name).stem
        slug = to_slug(title)
        wiki_path = settings.wiki_dir / f"{slug}.md"
        summary = wiki_path.read_text(encoding="utf-8")[:300] if wiki_path.exists() else ""
        summaries_parts.append(f"[{doc_id}] {title}: {summary}")

    materials_summaries = "\n".join(summaries_parts) or "(нет материалов)"

    system = "Ты — эксперт по корпоративной базе знаний BI AQYL."
    prompt = (
        (_PROMPTS_DIR / "search_expert.md")
        .read_text(encoding="utf-8")
        .format(
            query=body.query,
            materials_summaries=materials_summaries,
            language=body.language,
        )
    )

    llm = LLMClient()
    full_text = ""
    try:
        async for token in llm.stream_completion(system=system, prompt=prompt):
            full_text += token
            yield f"data: {json.dumps({'token': token})}\n\n"

        follow_ups = _parse_follow_ups(full_text)
        clean_answer = _strip_json_blocks(full_text)
        materials = [_fr_to_result(fr, score, _wiki_snippet(fr)) for fr, score in matched]
        done = {
            "done": True,
            "answer": clean_answer,
            "materials": [m.model_dump() for m in materials],
            "follow_ups": follow_ups,
        }
        yield f"data: {json.dumps(done)}\n\n"

    except Exception as exc:  # noqa: BLE001
        logger.error("expert_stream_error", error=str(exc))
        yield f'data: {json.dumps({"error": str(exc)})}\n\n'
    finally:
        await llm.aclose()


# ── advisor ───────────────────────────────────────────────────────────────────


def _build_chunks_text(hits: list[ChunkHit]) -> str:
    return "\n".join(f"[chunk_{i}] {h.text}" for i, h in enumerate(hits))


async def _advisor_stream(
    body: SearchRequest, session: AsyncSession
) -> AsyncGenerator[str, None]:
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
                "answer": "Контекст не содержит информации для ответа.",
                "contact": "knowledge-team@bi.group",
                "sources": [],
                "citations": [],
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

        # Citations: [chunk_N] patterns in the answer
        seen: set[str] = set()
        citations: list[dict[str, str]] = []
        for m in re.finditer(r'\[chunk_\d+\]', full_text):
            anchor = m.group(0)[1:-1]
            if anchor not in seen:
                seen.add(anchor)
                citations.append({"anchor": anchor})

        sources = [
            {"chunk_id": f"chunk_{i}", "document_id": h.slug, "text": h.text[:200]}
            for i, h in enumerate(hits)
        ]

        done = {
            "done": True,
            "answer": clean_answer,
            "sources": sources,
            "citations": citations,
            "follow_ups": follow_ups,
        }
        yield f"data: {json.dumps(done)}\n\n"

    except Exception as exc:  # noqa: BLE001
        logger.error("advisor_stream_error", error=str(exc))
        yield f'data: {json.dumps({"error": str(exc)})}\n\n'
    finally:
        await llm.aclose()
