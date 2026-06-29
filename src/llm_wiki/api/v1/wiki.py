"""Wiki browsing endpoints for the new Wiki tab."""

import re
from datetime import datetime, timezone

from fastapi import Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.api.deps import get_db
from llm_wiki.api.v1 import router
from llm_wiki.storage.wiki_fts import keyword_search


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
    slug: str, content: str, last_modified: float, snippet: str | None = None
) -> WikiPageSummary:
    """Build a WikiPageSummary from a slug + content + mtime."""
    return WikiPageSummary(
        slug=slug,
        title=_extract_title(content, slug),
        snippet=snippet if snippet is not None else _plain_snippet(content),
        size_chars=len(content),
        updated_at=datetime.fromtimestamp(last_modified, tz=timezone.utc).isoformat(),
        backlinks_count=content.count("[[") - content.count("[[]]"),
    )


@router.get("/wiki", response_model=list[WikiPageSummary])
async def list_wiki_pages(
    q: str | None = None,
    limit: int = 200,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
) -> list[WikiPageSummary]:
    """List wiki pages (newest first), or full-text search when *q* is given.

    With a query, runs the FTS lexical index over page **bodies** (ranked, with
    ``<mark>`` highlight snippets) — keyword search like Ctrl+F, used when the
    AI advisor is OFF. Without a query, lists all pages by recency.
    """
    from llm_wiki.storage.object_store import (
        WIKI_PREFIX,
        get_object_store,
        slug_from_wiki_key,
        wiki_key,
    )

    store = get_object_store()
    metas = {slug_from_wiki_key(m.key): m for m in store.list_objects(WIKI_PREFIX)}

    term = (q or "").strip()
    if term:
        hits = await keyword_search(db, term, limit=limit + offset)
        results: list[WikiPageSummary] = []
        for hit in hits:
            content = store.get_text(wiki_key(hit.slug))
            if content is None:
                continue
            meta = metas.get(hit.slug)
            mtime = meta.last_modified if meta else 0.0
            results.append(_summary(hit.slug, content, mtime, snippet=hit.snippet))
        return results[offset : offset + limit]

    ordered = sorted(metas.values(), key=lambda m: m.last_modified, reverse=True)
    results = []
    for meta in ordered:
        content = store.get_text(meta.key)
        if content is None:
            continue
        results.append(_summary(slug_from_wiki_key(meta.key), content, meta.last_modified))

    return results[offset : offset + limit]


@router.get("/wiki/{slug}/full", response_model=WikiPageDetail)
async def get_wiki_page_detail(slug: str) -> WikiPageDetail:
    """Get full markdown and backlinks for a wiki page."""
    from llm_wiki.storage.object_store import (
        WIKI_PREFIX,
        get_object_store,
        slug_from_wiki_key,
        wiki_key,
    )

    store = get_object_store()
    content = store.get_text(wiki_key(slug))
    if content is None:
        raise HTTPException(404, f"Wiki page '{slug}' not found")

    title = _extract_title(content, slug)

    backlinks: list[str] = []
    pattern = re.compile(rf"\[\[{re.escape(slug)}\]\]")
    mtime = 0.0
    for meta in store.list_objects(WIKI_PREFIX):
        other_slug = slug_from_wiki_key(meta.key)
        if other_slug == slug:
            mtime = meta.last_modified
            continue
        other = store.get_text(meta.key)
        if other and pattern.search(other):
            backlinks.append(other_slug)

    return WikiPageDetail(
        slug=slug,
        title=title,
        content=content,
        size_chars=len(content),
        updated_at=datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat(),
        backlinks=sorted(backlinks),
    )
