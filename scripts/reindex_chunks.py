"""Rebuild the ``chunks`` ChromaDB collection from all wiki pages.

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
        description="Rebuild the chunks ChromaDB collection from wiki pages."
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute chunks but do NOT write to ChromaDB or call the embeddings API.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=50,
        metavar="N",
        help="Print progress every N pages (default: 50).",
    )
    return p.parse_args()


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

    wiki_dir: Path = settings.wiki_dir
    if not wiki_dir.exists():
        print(f"Wiki directory does not exist: {wiki_dir}", file=sys.stderr)
        sys.exit(1)

    pages = sorted(wiki_dir.glob("*.md"))
    if not pages:
        print("No wiki pages found — nothing to index.")
        return

    print(f"Found {len(pages)} wiki page(s) in {wiki_dir}")

    # Dry-run: just count chunks without touching Chroma or the API
    if args.dry_run:
        total_chunks = 0
        for path in pages:
            content = path.read_text(encoding="utf-8")
            chunks = chunk_markdown(
                content, settings.chunk_max_chars, settings.chunk_overlap_chars
            )
            total_chunks += len(chunks)
            print(f"  {path.stem}: {len(chunks)} chunk(s)")
        print(f"\n[dry-run] Would index {total_chunks} chunks across {len(pages)} pages.")
        return

    # Real run
    llm = LLMClient()
    store = ChunkStore(chroma_path=settings.chroma_dir, llm_client=llm)

    print("Clearing existing chunks collection…")
    store.clear()

    indexed_pages = 0
    for i, path in enumerate(pages, start=1):
        slug = path.stem
        content = path.read_text(encoding="utf-8")
        title = _extract_title(content, slug)
        store.upsert_page(slug=slug, title=title, content=content, file_id="reindex-chunks")
        indexed_pages += 1

        if i % args.batch_size == 0 or i == len(pages):
            print(f"  Indexed {i}/{len(pages)} pages…")

    total_chunks = store.count()
    print(f"\nDone. {indexed_pages} pages → {total_chunks} chunks in ChromaDB.")
    print(
        "Tip: check usage.log for embedding costs:\n"
        "  docker compose exec api tail -n 20 data/usage.log | jq 'select(.agent_type==\"embed\")'"
    )


if __name__ == "__main__":
    main()
