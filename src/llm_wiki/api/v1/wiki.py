"""Wiki browsing endpoints for the Wiki tab (Postgres-backed)."""

import re
from datetime import datetime, timezone

from fastapi import Depends, HTTPException
from pydantic import BaseModel

from llm_wiki.api.deps import get_user_key
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
    sensitive: bool = False


class WikiPageDetail(BaseModel):
    """Full wiki page with backlinks."""

    slug: str
    title: str
    content: str
    size_chars: int
    updated_at: str
    backlinks: list[str]
    sensitive: bool = False


def _extract_title(content: str, slug: str, stored_title: str | None = None) -> str:
    """Display title for a page: the stored wiki title, else the body's first H1,
    else a humanised slug.

    Preferring the stored ``wiki_fts.title`` keeps uploaded-file pages from
    surfacing their ``private-<uuid>`` slug as a hash-like name (the slug is an
    opaque handle, not a title) and lets a source rename take effect without
    rewriting the page body. The humanised-slug branch is only ever reached for a
    legacy page that has neither a stored title nor an H1.
    """
    if stored_title and stored_title.strip():
        return stored_title.strip()
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
    slug: str,
    content: str,
    updated_at: datetime,
    snippet: str | None = None,
    sensitive: bool = False,
    stored_title: str | None = None,
) -> WikiPageSummary:
    """Build a WikiPageSummary from a slug + content + updated_at."""
    return WikiPageSummary(
        slug=slug,
        title=_extract_title(content, slug, stored_title),
        snippet=snippet if snippet is not None else _plain_snippet(content),
        size_chars=len(content),
        updated_at=updated_at.isoformat(),
        backlinks_count=content.count("[[") - content.count("[[]]"),
        sensitive=sensitive,
    )


@router.get("/wiki", response_model=list[WikiPageSummary])
async def list_wiki_pages(
    q: str | None = None,
    limit: int = 200,
    offset: int = 0,
    caller: str = Depends(get_user_key),
) -> list[WikiPageSummary]:
    """List wiki pages (newest first), or full-text search when *q* is given.

    With a query, runs the FTS lexical index over page **bodies** (ranked, with
    ``<mark>`` highlight snippets) — keyword search like Ctrl+F, used when the
    AI advisor is OFF. Without a query, lists all pages by recency.

    Private pages are only listed for their owner (``caller``).
    """
    term = (q or "").strip()
    if term:
        results: list[WikiPageSummary] = []
        for hit in wiki_store.keyword_search(term, limit=limit + offset, caller=caller):
            content = wiki_store.get_page(hit.slug, caller=caller)
            if content is None:
                continue
            meta = wiki_store.get_page_meta(hit.slug, caller=caller)
            updated = meta.updated_at if meta else datetime.now(timezone.utc)
            results.append(
                _summary(
                    hit.slug,
                    content,
                    updated,
                    snippet=hit.snippet,
                    sensitive=bool(meta and meta.sensitive),
                    stored_title=meta.title if meta else None,
                )
            )
        return results[offset : offset + limit]

    results = []
    for meta in wiki_store.list_pages(caller=caller):
        content = wiki_store.get_page(meta.slug, caller=caller)
        if content is None:
            continue
        results.append(
            _summary(
                meta.slug,
                content,
                meta.updated_at,
                sensitive=meta.sensitive,
                stored_title=meta.title,
            )
        )
    return results[offset : offset + limit]


@router.get("/wiki/{slug}/full", response_model=WikiPageDetail)
async def get_wiki_page_detail(
    slug: str,
    caller: str = Depends(get_user_key),
) -> WikiPageDetail:
    """Get full markdown and backlinks for a wiki page.

    A private page is returned only to its owner (``caller``); others get 404.
    """
    content = wiki_store.get_page(slug, caller=caller)
    if content is None:
        raise HTTPException(404, f"Wiki page '{slug}' not found")

    meta = wiki_store.get_page_meta(slug, caller=caller)
    title = _extract_title(content, slug, meta.title if meta else None)
    updated_at = meta.updated_at if meta else datetime.now(timezone.utc)

    backlinks: list[str] = []
    pattern = re.compile(rf"\[\[{re.escape(slug)}\]\]")
    for other_slug, other in wiki_store.get_all_pages(caller=caller):
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
        sensitive=bool(meta and meta.sensitive),
    )
