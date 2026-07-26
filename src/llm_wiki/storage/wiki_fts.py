"""PostgreSQL full-text index for wiki pages (LW-N4).

Lexical keyword search over ``title`` and ``body`` — no LLM, no embeddings.
Updated synchronously whenever the ingestion pipeline writes a wiki page.

Backed by a ``tsvector`` GIN index built with the ``russian`` text-search
config, which does proper Snowball stemming (so ``ценности`` matches
``ценностей`` natively) and emits ``<mark>`` highlight snippets via
``ts_headline``.
"""

from __future__ import annotations

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession

from dataclasses import dataclass

logger = structlog.get_logger(__name__)

# PostgreSQL text-search configuration (built-in Snowball stemmer).
_PG_TS_CONFIG = "russian"

# ts_headline options — emit <mark> highlights the frontend already renders.
_PG_HEADLINE_OPTS = (
    "StartSel=<mark>, StopSel=</mark>, MaxFragments=1, "
    "MaxWords=32, MinWords=12, ShortWord=2, HighlightAll=FALSE"
)

# An OR tsquery (recall-friendly): take websearch's AND query and swap & for |.
_PG_TSQUERY = (
    f"replace(websearch_to_tsquery('{_PG_TS_CONFIG}', :q)::text, '&', '|')::tsquery"
)

# A regular table; tsv is a STORED generated column (the 2-arg to_tsvector with
# an explicit config is IMMUTABLE, so this is allowed).
_TABLE_DDL = f"""
CREATE TABLE IF NOT EXISTS wiki_fts (
    slug  TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    body  TEXT NOT NULL DEFAULT '',
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    -- Privacy (LW private-wiki): a sensitive page is readable only by its owner.
    -- Public pages have sensitive=false / owner=NULL and behave as before.
    sensitive boolean NOT NULL DEFAULT false,
    owner text,
    tsv   tsvector GENERATED ALWAYS AS (
        to_tsvector('{_PG_TS_CONFIG}', coalesce(title, '') || ' ' || coalesce(body, ''))
    ) STORED
)
"""
_INDEX_DDL = "CREATE INDEX IF NOT EXISTS ix_wiki_fts_tsv ON wiki_fts USING GIN (tsv)"
# Add columns to tables created before these features landed (idempotent).
_ALTER_DDL = (
    "ALTER TABLE wiki_fts ADD COLUMN IF NOT EXISTS created_at timestamptz NOT NULL DEFAULT now()",
    "ALTER TABLE wiki_fts ADD COLUMN IF NOT EXISTS updated_at timestamptz NOT NULL DEFAULT now()",
    "ALTER TABLE wiki_fts ADD COLUMN IF NOT EXISTS sensitive boolean NOT NULL DEFAULT false",
    "ALTER TABLE wiki_fts ADD COLUMN IF NOT EXISTS owner text",
)


@dataclass(frozen=True)
class WikiFtsHit:
    """A single keyword-search result from ``wiki_fts``."""

    slug: str
    title: str
    snippet: str


def extract_page_title(content: str, slug: str) -> str:
    """Return the first H1 heading or a humanised slug."""
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return slug.replace("-", " ").title()


async def ensure_wiki_fts_table(conn: AsyncConnection) -> None:
    """Create the wiki table + GIN index (and backfill new columns) if needed."""
    await conn.execute(text(_TABLE_DDL))
    await conn.execute(text(_INDEX_DDL))
    for stmt in _ALTER_DDL:
        await conn.execute(text(stmt))


async def upsert_wiki_fts(
    session: AsyncSession,
    slug: str,
    title: str,
    body: str,
) -> None:
    """Insert or replace the indexed row for *slug*.

    Original casing is stored — ``to_tsvector`` normalises for matching while
    ``ts_headline`` uses the original text for readable snippets.
    """
    await session.execute(
        text(
            """
            INSERT INTO wiki_fts (slug, title, body)
            VALUES (:slug, :title, :body)
            ON CONFLICT (slug) DO UPDATE
                SET title = EXCLUDED.title, body = EXCLUDED.body
            """
        ),
        {"slug": slug, "title": title, "body": body},
    )
    await session.commit()
    logger.debug("wiki_fts_upserted", slug=slug)


async def delete_wiki_fts(session: AsyncSession, slug: str) -> None:
    """Remove a wiki page from the index."""
    await session.execute(text("DELETE FROM wiki_fts WHERE slug = :slug"), {"slug": slug})
    await session.commit()


async def keyword_search(
    session: AsyncSession,
    q: str,
    limit: int = 10,
) -> list[WikiFtsHit]:
    """Run a lexical full-text query and return ranked hits with snippets."""
    term = q.strip()
    if not term:
        return []

    result = await session.execute(
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
    )
    rows = result.fetchall()
    return [
        WikiFtsHit(slug=str(row[0]), title=str(row[1]), snippet=str(row[2]))
        for row in rows
    ]


async def rebuild_wiki_fts_from_store(session: AsyncSession) -> int:
    """Rebuild the entire index from every wiki page in the object store.

    Returns:
        Number of pages indexed.
    """
    from llm_wiki.storage.object_store import (
        WIKI_PREFIX,
        get_object_store,
        slug_from_wiki_key,
    )

    store = get_object_store()
    await session.execute(text("DELETE FROM wiki_fts"))
    await session.commit()
    count = 0
    for obj in store.list_objects(WIKI_PREFIX):
        content = store.get_text(obj.key)
        if content is None:
            continue
        slug = slug_from_wiki_key(obj.key)
        title = extract_page_title(content, slug)
        await upsert_wiki_fts(session, slug, title, content)
        count += 1
    logger.info("wiki_fts_rebuilt", pages=count)
    return count


async def wiki_fts_count(session: AsyncSession) -> int:
    """Return the number of rows in ``wiki_fts``."""
    result = await session.execute(text("SELECT COUNT(*) FROM wiki_fts"))
    return int(result.scalar_one() or 0)
