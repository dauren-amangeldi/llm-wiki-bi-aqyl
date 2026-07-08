"""Wiki browsing endpoints for the Wiki tab (Postgres-backed)."""

import re
from datetime import datetime, timezone

from fastapi import HTTPException
from pydantic import BaseModel

from llm_wiki.api.v1 import router
from llm_wiki.storage import wiki_store


class WikiPageSummary(BaseModel):
    """Summary row for the wiki page list."""

    slug: str
    title: str
    snippet: str
    size_chars: int
    updated_at: str
    backlinks_count: int


class WikiPageDetail(BaseModel):
    """Full wiki page with backlinks."""

    slug: str
    title: str
    content: str
    size_chars: int
    updated_at: str
    backlinks: list[str]


def _extract_title(content: str, slug: str) -> str:
    """Return the first H1 heading or a humanised slug."""
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return slug.replace("-", " ").title()


def _plain_snippet(content: str, length: int = 200) -> str:
    """Strip markdown noise and return a short plain-text preview."""
    text = re.sub(r"^#+\s*", "", content, flags=re.MULTILINE)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"[*_`>]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:length] + ("…" if len(text) > length else "")


def _summary(
    slug: str, content: str, updated_at: datetime, snippet: str | None = None
) -> WikiPageSummary:
    """Build a WikiPageSummary from a slug + content + updated_at."""
    return WikiPageSummary(
        slug=slug,
        title=_extract_title(content, slug),
        snippet=snippet if snippet is not None else _plain_snippet(content),
        size_chars=len(content),
        updated_at=updated_at.isoformat(),
        backlinks_count=content.count("[[") - content.count("[[]]"),
    )


@router.get("/wiki", response_model=list[WikiPageSummary])
async def list_wiki_pages(
    q: str | None = None,
    limit: int = 200,
    offset: int = 0,
) -> list[WikiPageSummary]:
    """List wiki pages (newest first), or full-text search when *q* is given.

    With a query, runs the FTS lexical index over page **bodies** (ranked, with
    ``<mark>`` highlight snippets) — keyword search like Ctrl+F, used when the
    AI advisor is OFF. Without a query, lists all pages by recency.
    """
    term = (q or "").strip()
    if term:
        results: list[WikiPageSummary] = []
        for hit in wiki_store.keyword_search(term, limit=limit + offset):
            content = wiki_store.get_page(hit.slug)
            if content is None:
                continue
            meta = wiki_store.get_page_meta(hit.slug)
            updated = meta.updated_at if meta else datetime.now(timezone.utc)
            results.append(_summary(hit.slug, content, updated, snippet=hit.snippet))
        return results[offset : offset + limit]

    results = []
    for meta in wiki_store.list_pages():
        content = wiki_store.get_page(meta.slug)
        if content is None:
            continue
        results.append(_summary(meta.slug, content, meta.updated_at))
    return results[offset : offset + limit]


@router.get("/wiki/{slug}/full", response_model=WikiPageDetail)
async def get_wiki_page_detail(slug: str) -> WikiPageDetail:
    """Get full markdown and backlinks for a wiki page."""
    content = wiki_store.get_page(slug)
    if content is None:
        raise HTTPException(404, f"Wiki page '{slug}' not found")

    title = _extract_title(content, slug)
    meta = wiki_store.get_page_meta(slug)
    updated_at = meta.updated_at if meta else datetime.now(timezone.utc)

    backlinks: list[str] = []
    pattern = re.compile(rf"\[\[{re.escape(slug)}\]\]")
    for other_slug, other in wiki_store.get_all_pages():
        if other_slug == slug:
            continue
        if other and pattern.search(other):
            backlinks.append(other_slug)

    return WikiPageDetail(
        slug=slug,
        title=title,
        content=content,
        size_chars=len(content),
        updated_at=updated_at.isoformat(),
        backlinks=sorted(backlinks),
    )
