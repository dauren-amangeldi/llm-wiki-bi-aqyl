"""SQLite FTS5 full-text index for wiki pages (LW-N4).

Lexical keyword search over ``slug``, ``title``, and ``body`` — no LLM,
no embeddings.  Updated synchronously whenever the ingestion pipeline
writes a wiki page to disk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession

logger = structlog.get_logger(__name__)

_FTS_DDL = """
CREATE VIRTUAL TABLE IF NOT EXISTS wiki_fts USING fts5(
    slug UNINDEXED,
    title,
    body,
    tokenize='unicode61'
)
"""


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


def _prepare_fts_query(q: str) -> str:
    """Turn a user query into a safe FTS5 MATCH expression (prefix AND)."""
    tokens = re.findall(r"[\w\u0400-\u04ff]+", q, flags=re.UNICODE)
    if not tokens:
        return ""
    parts: list[str] = []
    for token in tokens:
        safe = token.replace('"', '""')
        parts.append(f'"{safe}"*')
    return " AND ".join(parts)


async def ensure_wiki_fts_table(conn: AsyncConnection) -> None:
    """Create the ``wiki_fts`` virtual table if it does not exist."""
    await conn.execute(text(_FTS_DDL))


async def upsert_wiki_fts(
    session: AsyncSession,
    slug: str,
    title: str,
    body: str,
) -> None:
    """Replace the FTS row for *slug* (delete + insert)."""
    await session.execute(text("DELETE FROM wiki_fts WHERE slug = :slug"), {"slug": slug})
    await session.execute(
        text("INSERT INTO wiki_fts(slug, title, body) VALUES (:slug, :title, :body)"),
        {"slug": slug, "title": title, "body": body},
    )
    await session.commit()
    logger.debug("wiki_fts_upserted", slug=slug)


async def delete_wiki_fts(session: AsyncSession, slug: str) -> None:
    """Remove a wiki page from the FTS index."""
    await session.execute(text("DELETE FROM wiki_fts WHERE slug = :slug"), {"slug": slug})
    await session.commit()


async def keyword_search(
    session: AsyncSession,
    q: str,
    limit: int = 10,
) -> list[WikiFtsHit]:
    """Run a lexical FTS5 query and return ranked hits with snippets."""
    fts_q = _prepare_fts_query(q.strip())
    if not fts_q:
        return []

    result = await session.execute(
        text(
            """
            SELECT slug, title,
                   snippet(wiki_fts, 2, '<mark>', '</mark>', '…', 32) AS snippet
            FROM wiki_fts
            WHERE wiki_fts MATCH :q
            ORDER BY rank
            LIMIT :limit
            """
        ),
        {"q": fts_q, "limit": limit},
    )
    rows = result.fetchall()
    return [
        WikiFtsHit(slug=str(row[0]), title=str(row[1]), snippet=str(row[2]))
        for row in rows
    ]


async def rebuild_wiki_fts_from_disk(
    session: AsyncSession,
    wiki_dir: Path,
) -> int:
    """Rebuild the entire FTS index from ``wiki_dir/*.md``.

    Returns:
        Number of pages indexed.
    """
    if not wiki_dir.exists():
        return 0

    await session.execute(text("DELETE FROM wiki_fts"))
    count = 0
    for path in sorted(wiki_dir.glob("*.md")):
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            continue
        slug = path.stem
        title = extract_page_title(content, slug)
        await session.execute(
            text("INSERT INTO wiki_fts(slug, title, body) VALUES (:slug, :title, :body)"),
            {"slug": slug, "title": title, "body": content},
        )
        count += 1
    await session.commit()
    logger.info("wiki_fts_rebuilt", pages=count)
    return count


async def wiki_fts_count(session: AsyncSession) -> int:
    """Return the number of rows in ``wiki_fts``."""
    result = await session.execute(text("SELECT COUNT(*) FROM wiki_fts"))
    return int(result.scalar_one() or 0)
