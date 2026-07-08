"""PostgreSQL-backed wiki page store — the source of truth for wiki pages.

Replaces the former ``wiki/{slug}.md`` objects in S3: page content now lives in
the ``wiki_fts`` table (slug / title / body + a generated ``tsvector``), so the
app pods stay stateless and S3 holds only raw uploads.

Synchronous accessors (via the shared sync engine, like the vector stores) so
both sync callers (pipeline, agents, backlinks) and async callers (API routes)
use one uniform API. Wiki reads are fast single-row/GIN queries on local
Postgres, so calling them from an async handler is fine (same pattern as the
former synchronous object-store reads).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import structlog
from sqlalchemy import text

from llm_wiki.storage.metadata import get_sync_engine
from llm_wiki.storage.wiki_fts import (
    WikiFtsHit,
    _PG_HEADLINE_OPTS,
    _PG_TS_CONFIG,
    _PG_TSQUERY,
    extract_page_title,
)

logger = structlog.get_logger(__name__)

__all__ = [
    "WikiFtsHit",
    "WikiPageMeta",
    "extract_page_title",
    "save_page",
    "get_page",
    "get_page_title",
    "page_exists",
    "list_pages",
    "get_all_pages",
    "delete_page",
    "count",
    "keyword_search",
]


@dataclass(frozen=True)
class WikiPageMeta:
    """Lightweight metadata for a wiki page (no body)."""

    slug: str
    title: str
    updated_at: datetime


def save_page(slug: str, title: str, body: str) -> None:
    """Insert or replace a wiki page. ``updated_at`` is bumped; ``created_at`` sticks."""
    now = datetime.now(timezone.utc)
    with get_sync_engine().begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO wiki_fts (slug, title, body, created_at, updated_at)
                VALUES (:slug, :title, :body, :now, :now)
                ON CONFLICT (slug) DO UPDATE
                    SET title = EXCLUDED.title,
                        body = EXCLUDED.body,
                        updated_at = EXCLUDED.updated_at
                """
            ),
            {"slug": slug, "title": title, "body": body, "now": now},
        )
    logger.debug("wiki_page_saved", slug=slug)


def get_page(slug: str) -> str | None:
    """Return the page body for *slug*, or None if it does not exist."""
    with get_sync_engine().connect() as conn:
        row = conn.execute(
            text("SELECT body FROM wiki_fts WHERE slug = :slug"), {"slug": slug}
        ).first()
    return None if row is None else str(row[0])


def get_page_title(slug: str) -> str | None:
    """Return the page title for *slug*, or None if it does not exist."""
    with get_sync_engine().connect() as conn:
        row = conn.execute(
            text("SELECT title FROM wiki_fts WHERE slug = :slug"), {"slug": slug}
        ).first()
    return None if row is None else str(row[0])


def get_page_meta(slug: str) -> WikiPageMeta | None:
    """Return slug/title/updated_at for *slug*, or None if it does not exist."""
    with get_sync_engine().connect() as conn:
        row = conn.execute(
            text("SELECT slug, title, updated_at FROM wiki_fts WHERE slug = :slug"),
            {"slug": slug},
        ).first()
    return (
        None
        if row is None
        else WikiPageMeta(slug=str(row[0]), title=str(row[1]), updated_at=row[2])
    )


def page_exists(slug: str) -> bool:
    """Return True if a page with *slug* exists."""
    with get_sync_engine().connect() as conn:
        row = conn.execute(
            text("SELECT 1 FROM wiki_fts WHERE slug = :slug"), {"slug": slug}
        ).first()
    return row is not None


def list_pages() -> list[WikiPageMeta]:
    """Return metadata for all pages, newest-updated first."""
    with get_sync_engine().connect() as conn:
        rows = conn.execute(
            text(
                "SELECT slug, title, updated_at FROM wiki_fts "
                "ORDER BY updated_at DESC NULLS LAST, slug"
            )
        ).all()
    return [
        WikiPageMeta(slug=str(r[0]), title=str(r[1]), updated_at=r[2])
        for r in rows
    ]


def get_all_pages() -> list[tuple[str, str]]:
    """Return ``(slug, body)`` for every page (for linter / auditor / rebuilds)."""
    with get_sync_engine().connect() as conn:
        rows = conn.execute(
            text("SELECT slug, body FROM wiki_fts ORDER BY slug")
        ).all()
    return [(str(r[0]), str(r[1])) for r in rows]


def delete_page(slug: str) -> None:
    """Remove a page. No-op if absent."""
    with get_sync_engine().begin() as conn:
        conn.execute(text("DELETE FROM wiki_fts WHERE slug = :slug"), {"slug": slug})
    logger.debug("wiki_page_deleted", slug=slug)


def count() -> int:
    """Return the number of wiki pages."""
    with get_sync_engine().connect() as conn:
        return int(conn.execute(text("SELECT COUNT(*) FROM wiki_fts")).scalar_one() or 0)


def keyword_search(q: str, limit: int = 10) -> list[WikiFtsHit]:
    """Lexical full-text search over title+body, with ``<mark>`` snippets."""
    term = q.strip()
    if not term:
        return []
    with get_sync_engine().connect() as conn:
        rows = conn.execute(
            text(
                f"""
                WITH query AS (SELECT {_PG_TSQUERY} AS tsq)
                SELECT slug, title,
                       ts_headline('{_PG_TS_CONFIG}', body, query.tsq,
                                   '{_PG_HEADLINE_OPTS}') AS snippet
                FROM wiki_fts, query
                WHERE tsv @@ query.tsq
                ORDER BY ts_rank(tsv, query.tsq) DESC
                LIMIT :limit
                """
            ),
            {"q": term, "limit": limit},
        ).all()
    return [
        WikiFtsHit(slug=str(r[0]), title=str(r[1]), snippet=str(r[2])) for r in rows
    ]
