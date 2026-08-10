"""PostgreSQL-backed wiki page store — the source of truth for wiki pages.

Replaces the former ``wiki/{slug}.md`` objects in S3: page content now lives in
the ``wiki_fts`` table (slug / title / body + a generated ``tsvector``), so the
app pods stay stateless and S3 holds only raw uploads.

Synchronous accessors (via the shared sync engine, like the vector stores) so
both sync callers (pipeline, agents, backlinks) and async callers (API routes)
use one uniform API. Wiki reads are fast single-row/GIN queries on local
Postgres, so calling them from an async handler is fine (same pattern as the
former synchronous object-store reads).

Privacy (private wiki): a page can be marked ``sensitive`` with an ``owner``.
Such a page is a normal wiki page for its owner (readable, searchable) but is
invisible to everyone else. Two visibility rules keep this safe:

* **Enumerations** (``list_pages`` / ``keyword_search`` / ``get_all_pages``)
  default to *public-only* when no ``caller`` is given. Internal bulk callers
  (linter, auditor, backlink scans) pass no caller and therefore can never pull
  private content into a shared report. A concrete caller also sees their own
  private pages.
* **Point lookups** (``get_page`` / ``get_page_meta`` / ``get_page_title`` /
  ``page_exists``) are unfiltered when ``caller`` is None (trusted internal
  slug-based access — the slug is already known and not exposed cross-user) and
  enforce owner-only access when a ``caller`` is supplied. Every user-facing
  endpoint passes the caller so a non-owner gets ``None`` (→ 404) for a
  private slug.
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
    "set_pages_visibility",
    "set_pages_title",
    "get_page",
    "get_page_title",
    "get_page_meta",
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
    sensitive: bool = False


def _visibility(caller: str | None) -> tuple[str, dict[str, object]]:
    """SQL predicate + params limiting an *enumeration* to visible pages.

    ``caller is None`` → public pages only (safe default for internal bulk
    callers like the linter/auditor/backlink scan, so private content never
    leaks into a shared report). A concrete caller also sees their own private
    pages.
    """
    if caller is None:
        return "NOT sensitive", {}
    return "(NOT sensitive OR owner = :caller)", {"caller": caller}


def _visible_to(caller: str | None, sensitive: object, owner: object) -> bool:
    """Whether a *point-looked-up* row is visible to *caller*.

    Unfiltered for internal callers (``caller is None``); owner-only when a
    concrete caller is supplied.
    """
    if caller is None:
        return True
    return (not bool(sensitive)) or owner == caller


def save_page(
    slug: str,
    title: str,
    body: str,
    sensitive: bool = False,
    owner: str | None = None,
) -> None:
    """Insert or replace a wiki page. ``updated_at`` is bumped; ``created_at`` sticks.

    ``sensitive`` + ``owner`` mark a private, owner-only page.
    """
    now = datetime.now(timezone.utc)
    with get_sync_engine().begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO wiki_fts (slug, title, body, sensitive, owner, created_at, updated_at)
                VALUES (:slug, :title, :body, :sensitive, :owner, :now, :now)
                ON CONFLICT (slug) DO UPDATE
                    SET title = EXCLUDED.title,
                        body = EXCLUDED.body,
                        sensitive = EXCLUDED.sensitive,
                        owner = EXCLUDED.owner,
                        updated_at = EXCLUDED.updated_at
                """
            ),
            {
                "slug": slug,
                "title": title,
                "body": body,
                "sensitive": sensitive,
                "owner": owner,
                "now": now,
            },
        )
    logger.debug("wiki_page_saved", slug=slug, sensitive=sensitive)


def set_pages_visibility(
    slugs: list[str], *, sensitive: bool, owner: str | None
) -> None:
    """Flip ``sensitive`` / ``owner`` for a set of pages in one statement.

    Used by the case-publish cascade: a case is the single source of truth for
    the privacy of its nested materials, so when it is published (or made
    private) every wiki page it owns follows. No-op for an empty slug list.
    """
    slugs = [s for s in slugs if s]
    if not slugs:
        return
    now = datetime.now(timezone.utc)
    with get_sync_engine().begin() as conn:
        conn.execute(
            text(
                "UPDATE wiki_fts SET sensitive = :sensitive, owner = :owner, "
                "updated_at = :now WHERE slug = ANY(:slugs)"
            ),
            {"sensitive": sensitive, "owner": owner, "now": now, "slugs": slugs},
        )
    logger.debug("wiki_pages_visibility_set", count=len(slugs), sensitive=sensitive)


def set_pages_title(slugs: list[str], title: str) -> None:
    """Set the stored ``title`` for a set of pages in one statement.

    Used when a source file is renamed: its own wiki page(s) follow so the
    reader header and citation footer show the new name. The page body's H1 is
    left untouched — the display title now prefers this stored value (see the
    wiki API's title resolution), so a rename takes effect without rewriting the
    body. No-op for an empty slug list or a blank title.
    """
    slugs = [s for s in slugs if s]
    title = (title or "").strip()
    if not slugs or not title:
        return
    now = datetime.now(timezone.utc)
    with get_sync_engine().begin() as conn:
        conn.execute(
            text(
                "UPDATE wiki_fts SET title = :title, updated_at = :now "
                "WHERE slug = ANY(:slugs)"
            ),
            {"title": title, "now": now, "slugs": slugs},
        )
    logger.debug("wiki_pages_title_set", count=len(slugs), title=title)


def get_page(slug: str, caller: str | None = None) -> str | None:
    """Return the page body for *slug*, or None if absent / not visible to *caller*."""
    with get_sync_engine().connect() as conn:
        row = conn.execute(
            text("SELECT body, sensitive, owner FROM wiki_fts WHERE slug = :slug"),
            {"slug": slug},
        ).first()
    if row is None or not _visible_to(caller, row[1], row[2]):
        return None
    return str(row[0])


def get_page_title(slug: str, caller: str | None = None) -> str | None:
    """Return the page title for *slug*, or None if absent / not visible to *caller*."""
    with get_sync_engine().connect() as conn:
        row = conn.execute(
            text("SELECT title, sensitive, owner FROM wiki_fts WHERE slug = :slug"),
            {"slug": slug},
        ).first()
    if row is None or not _visible_to(caller, row[1], row[2]):
        return None
    return str(row[0])


def get_page_meta(slug: str, caller: str | None = None) -> WikiPageMeta | None:
    """Return slug/title/updated_at/sensitive for *slug*, or None if absent / hidden."""
    with get_sync_engine().connect() as conn:
        row = conn.execute(
            text(
                "SELECT slug, title, updated_at, sensitive, owner "
                "FROM wiki_fts WHERE slug = :slug"
            ),
            {"slug": slug},
        ).first()
    if row is None or not _visible_to(caller, row[3], row[4]):
        return None
    return WikiPageMeta(
        slug=str(row[0]), title=str(row[1]), updated_at=row[2], sensitive=bool(row[3])
    )


def page_exists(slug: str, caller: str | None = None) -> bool:
    """Return True if a page with *slug* exists and is visible to *caller*."""
    with get_sync_engine().connect() as conn:
        row = conn.execute(
            text("SELECT sensitive, owner FROM wiki_fts WHERE slug = :slug"),
            {"slug": slug},
        ).first()
    return row is not None and _visible_to(caller, row[0], row[1])


def list_pages(caller: str | None = None) -> list[WikiPageMeta]:
    """Return metadata for visible pages, newest-updated first."""
    clause, params = _visibility(caller)
    with get_sync_engine().connect() as conn:
        rows = conn.execute(
            text(
                "SELECT slug, title, updated_at, sensitive FROM wiki_fts "
                f"WHERE {clause} "
                "ORDER BY updated_at DESC NULLS LAST, slug"
            ),
            params,
        ).all()
    return [
        WikiPageMeta(
            slug=str(r[0]), title=str(r[1]), updated_at=r[2], sensitive=bool(r[3])
        )
        for r in rows
    ]


def get_all_pages(caller: str | None = None) -> list[tuple[str, str]]:
    """Return ``(slug, body)`` for every visible page (for linter / auditor / rebuilds).

    Defaults to public pages only so bulk quality/audit passes never read
    private content; pass a *caller* to also include that user's private pages.
    """
    clause, params = _visibility(caller)
    with get_sync_engine().connect() as conn:
        rows = conn.execute(
            text(f"SELECT slug, body FROM wiki_fts WHERE {clause} ORDER BY slug"),
            params,
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


def keyword_search(q: str, limit: int = 10, caller: str | None = None) -> list[WikiFtsHit]:
    """Lexical full-text search over title+body, with ``<mark>`` snippets.

    Respects page privacy: without a *caller* only public pages match; with a
    caller their own private pages are included too.
    """
    term = q.strip()
    if not term:
        return []
    clause, extra = _visibility(caller)
    with get_sync_engine().connect() as conn:
        rows = conn.execute(
            text(
                f"""
                WITH query AS (SELECT {_PG_TSQUERY} AS tsq)
                SELECT slug, title,
                       ts_headline('{_PG_TS_CONFIG}', body, query.tsq,
                                   '{_PG_HEADLINE_OPTS}') AS snippet
                FROM wiki_fts, query
                WHERE tsv @@ query.tsq AND {clause}
                ORDER BY ts_rank(tsv, query.tsq) DESC
                LIMIT :limit
                """
            ),
            {"q": term, "limit": limit, **extra},
        ).all()
    if not rows:
        # No lexical hit — fall back to trigram similarity on the title so a
        # typo (e.g. «маркетнг») still finds «Маркетинг…». Snippet is a plain
        # body prefix (no <mark>); HitSnippet renders it fine.
        with get_sync_engine().connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT slug, title, left(body, 200) AS snippet FROM wiki_fts "
                    f"WHERE {clause} AND word_similarity(lower(:q), lower(title)) > 0.3 "
                    "ORDER BY word_similarity(lower(:q), lower(title)) DESC LIMIT :limit"
                ),
                {"q": term, "limit": limit, **extra},
            ).all()
    return [
        WikiFtsHit(slug=str(r[0]), title=str(r[1]), snippet=str(r[2])) for r in rows
    ]
