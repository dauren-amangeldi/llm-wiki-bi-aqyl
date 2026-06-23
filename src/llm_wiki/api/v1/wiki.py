"""Wiki browsing endpoints for the new Wiki tab."""

import re
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from llm_wiki.api.deps import get_db
from llm_wiki.api.v1 import router
from llm_wiki.config import settings
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


def _summary_from_path(path: Path, snippet: str | None = None) -> WikiPageSummary | None:
    """Build a WikiPageSummary from a wiki .md file (snippet overridable)."""
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return None
    slug = path.stem
    return WikiPageSummary(
        slug=slug,
        title=_extract_title(content, slug),
        snippet=snippet if snippet is not None else _plain_snippet(content),
        size_chars=len(content),
        updated_at=datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
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

    With a query, runs the FTS5 lexical index over page **bodies** (ranked,
    with ``<mark>`` highlight snippets) — keyword search like Ctrl+F, used when
    the AI advisor is OFF. Without a query, lists all pages by recency.
    """
    wiki_dir = settings.wiki_dir
    if not wiki_dir.exists():
        return []

    term = (q or "").strip()
    if term:
        hits = await keyword_search(db, term, limit=limit + offset)
        results: list[WikiPageSummary] = []
        for hit in hits:
            summary = _summary_from_path(wiki_dir / f"{hit.slug}.md", snippet=hit.snippet)
            if summary is not None:
                results.append(summary)
        return results[offset : offset + limit]

    files = sorted(
        wiki_dir.glob("*.md"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    results: list[WikiPageSummary] = []
    for path in files:
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            continue
        slug = path.stem
        title = _extract_title(content, slug)
        if q and q.lower() not in title.lower() and q.lower() not in content.lower():
            continue
        results.append(
            WikiPageSummary(
                slug=slug,
                title=title,
                snippet=_plain_snippet(content),
                size_chars=len(content),
                updated_at=datetime.fromtimestamp(
                    path.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
                backlinks_count=content.lower().count("[[")
                - content.lower().count("[[]]"),
            )
        )

    return results[offset : offset + limit]


@router.get("/wiki/{slug}/full", response_model=WikiPageDetail)
async def get_wiki_page_detail(slug: str) -> WikiPageDetail:
    """Get full markdown and backlinks for a wiki page."""
    wiki_dir = settings.wiki_dir
    path = wiki_dir / f"{slug}.md"
    if not path.exists():
        raise HTTPException(404, f"Wiki page '{slug}' not found")

    content = path.read_text(encoding="utf-8")
    title = _extract_title(content, slug)

    backlinks: list[str] = []
    pattern = re.compile(rf"\[\[{re.escape(slug)}\]\]")
    for other in wiki_dir.glob("*.md"):
        if other.stem == slug:
            continue
        try:
            if pattern.search(other.read_text(encoding="utf-8")):
                backlinks.append(other.stem)
        except OSError:
            continue

    return WikiPageDetail(
        slug=slug,
        title=title,
        content=content,
        size_chars=len(content),
        updated_at=datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
        backlinks=sorted(backlinks),
    )
