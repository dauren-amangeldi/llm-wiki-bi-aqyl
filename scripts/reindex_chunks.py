"""Rebuild the ``chunks`` pgvector collection from all wiki pages.

Run this:
  - On first deploy (after wiki pages already exist but chunks collection is empty).
  - After changing EMBEDDING_MODEL or EMBEDDING_DIMENSIONS.
  - After a bulk import that bypassed the normal ingestion pipeline.

Usage:
    docker compose exec api uv run python scripts/reindex_chunks.py
    docker compose exec api uv run python scripts/reindex_chunks.py --dry-run
    docker compose exec api uv run python scripts/reindex_chunks.py --batch-size 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is on the path when run directly outside Docker
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rebuild the chunks pgvector collection from wiki pages."
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute chunks but do NOT write to pgvector or call the embeddings API.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=50,
        metavar="N",
        help="Print progress every N pages (default: 50).",
    )
    p.add_argument(
        "--backfill-file-ids",
        action="store_true",
        help="Only backfill file_id metadata on existing chunks (no re-embed).",
    )
    return p.parse_args()


def _load_slug_file_map() -> dict[str, str]:
    """Load slug → file_id mapping from SQLite (sync wrapper for CLI scripts)."""
    import asyncio

    from llm_wiki.api.deps import _SessionLocal
    from llm_wiki.storage.metadata import build_slug_to_file_id_map

    async def _run() -> dict[str, str]:
        async with _SessionLocal() as session:
            return await build_slug_to_file_id_map(session)

    return asyncio.run(_run())


def _extract_title(content: str, slug: str) -> str:
    """Return the first # heading from *content*, or the slug as fallback."""
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return slug.replace("-", " ").title()


def main() -> None:
    args = _parse_args()

    from llm_wiki.config import settings
    from llm_wiki.llm.chunk_store import ChunkStore, chunk_markdown
    from llm_wiki.llm.client import LLMClient

    if args.backfill_file_ids:
        slug_map = _load_slug_file_map()
        if not slug_map:
            print("No slug→file_id mappings found in metadata.db.")
            return
        llm = LLMClient()
        store = ChunkStore(llm_client=llm)
        total = sum(store.backfill_file_id(slug, fid) for slug, fid in slug_map.items())
        print(f"Backfill complete: {total} chunk record(s) updated.")
        return

    from llm_wiki.storage import wiki_store

    pages = wiki_store.get_all_pages()  # list[(slug, content)] from Postgres
    if not pages:
        print("No wiki pages found — nothing to index.")
        return

    print(f"Found {len(pages)} wiki page(s) in Postgres")

    # Dry-run: just count chunks without touching pgvector or the API
    if args.dry_run:
        total_chunks = 0
        for slug, content in pages:
            chunks = chunk_markdown(
                content, settings.chunk_max_chars, settings.chunk_overlap_chars
            )
            total_chunks += len(chunks)
            print(f"  {slug}: {len(chunks)} chunk(s)")
        print(f"\n[dry-run] Would index {total_chunks} chunks across {len(pages)} pages.")
        return

    # Real run
    llm = LLMClient()
    store = ChunkStore(llm_client=llm)

    print("Clearing existing chunks…")
    store.clear()

    slug_map = _load_slug_file_map()
    indexed_pages = 0
    for i, (slug, content) in enumerate(pages, start=1):
        title = _extract_title(content, slug)
        source_file_id = slug_map.get(slug, "reindex-chunks")
        store.upsert_page(
            slug=slug, title=title, content=content, file_id=source_file_id
        )
        indexed_pages += 1

        if i % args.batch_size == 0 or i == len(pages):
            print(f"  Indexed {i}/{len(pages)} pages…")

    total_chunks = store.count()
    print(f"\nDone. {indexed_pages} pages → {total_chunks} chunks in pgvector.")
    print(
        "Tip: check usage.log for embedding costs:\n"
        "  docker compose exec api tail -n 20 data/usage.log | jq 'select(.agent_type==\"embed\")'"
    )


if __name__ == "__main__":
    main()
