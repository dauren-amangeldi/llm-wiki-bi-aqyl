"""Rebuild the pgvector heading index from the current data/index.md.

Run inside the container after restoring a backup or after switching models:
    docker compose exec api uv run python scripts/reindex.py

Use --dry-run to preview what would be indexed without making any changes.
Exit code 1 on any error.
"""

import argparse
import sys
import time
from pathlib import Path

# Resolve the src/ package root so this script works both inside and outside Docker
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import structlog

logger = structlog.get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild pgvector heading index from index.md",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview headings without writing to pgvector.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Override EMBEDDING_BATCH_SIZE from config (0 = use config value).",
    )
    parser.add_argument(
        "--backfill-file-ids",
        action="store_true",
        help="Only backfill file_id metadata on existing headings/chunks (no re-embed).",
    )
    return parser.parse_args()


def _load_slug_file_map() -> dict[str, str]:
    """Load slug → file_id mapping from SQLite (sync wrapper for CLI scripts)."""
    import asyncio

    from llm_wiki.api.deps import _SessionLocal
    from llm_wiki.storage.metadata import build_slug_to_file_id_map

    async def _run() -> dict[str, str]:
        async with _SessionLocal() as session:
            return await build_slug_to_file_id_map(session)

    return asyncio.run(_run())


def _backfill_file_ids_only() -> None:
    """Backfill file_id on existing pgvector metadata without re-embedding."""
    from llm_wiki.config import settings
    from llm_wiki.llm.chunk_store import ChunkStore
    from llm_wiki.llm.client import LLMClient
    from llm_wiki.llm.embeddings import EmbeddingStore

    slug_map = _load_slug_file_map()
    if not slug_map:
        print("No slug→file_id mappings found in metadata.db.")
        return

    llm = LLMClient()
    heading_store = EmbeddingStore(llm_client=llm)
    chunk_store = ChunkStore(llm_client=llm)

    heading_updates = 0
    chunk_updates = 0
    for slug, file_id in slug_map.items():
        if heading_store.backfill_file_id(slug, file_id):
            heading_updates += 1
        chunk_updates += chunk_store.backfill_file_id(slug, file_id)

    print(
        f"Backfill complete: {heading_updates} heading(s), "
        f"{chunk_updates} chunk record(s) updated."
    )


def main() -> None:
    """Scan index.md and re-embed all headings into pgvector."""
    args = _parse_args()

    if args.backfill_file_ids:
        _backfill_file_ids_only()
        return

    from llm_wiki.config import settings
    from llm_wiki.llm.client import LLMClient
    from llm_wiki.llm.embeddings import EmbeddingStore, HeadingInfo
    from llm_wiki.storage.index import IndexStorage

    if not settings.index_path.exists():
        print(f"ERROR: index.md not found at {settings.index_path}", file=sys.stderr)
        sys.exit(1)

    if not settings.openai_api_key:
        print(
            "ERROR: OPENAI_API_KEY is required for embeddings. Set it in your .env file.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Parse all headings from index.md
    index_storage = IndexStorage(settings.index_path)
    raw_headings = index_storage.read_headings()

    # Build HeadingInfo list from parsed headings (only those with a slug)
    heading_infos: list[HeadingInfo] = []
    current_section = "General"
    for h in raw_headings:
        if h.level == 2 and h.slug is None:
            current_section = h.text
        if h.slug:
            heading_infos.append(
                HeadingInfo(
                    slug=h.slug,
                    title=h.text.replace(f"[[{h.slug}]]", "").strip(" —") or h.slug,
                    section=current_section,
                    level=h.level,
                )
            )

    print(f"Found {len(heading_infos)} headings in {settings.index_path}")

    if args.dry_run:
        print("\n--- DRY RUN — no changes written ---")
        for hi in heading_infos:
            print(f"  [{hi.level}] {hi.slug!r:40s}  section={hi.section!r}")
        print(f"\nTotal: {len(heading_infos)} headings would be indexed.")
        return

    if not heading_infos:
        print("Nothing to index. Exiting.")
        return

    if args.batch_size > 0:
        settings.__dict__["embedding_batch_size"] = args.batch_size  # runtime override

    llm = LLMClient()
    store = EmbeddingStore(llm_client=llm)

    print(f"Clearing existing collection …")
    store.clear()

    print(f"Embedding {len(heading_infos)} headings (model={settings.embedding_model}) …")
    t0 = time.perf_counter()

    slug_map = _load_slug_file_map()

    try:
        store.upsert_many(heading_infos, slug_to_file_id=slug_map)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR during embedding: {exc}", file=sys.stderr)
        sys.exit(1)

    elapsed = time.perf_counter() - t0
    final_count = store.count()

    print(
        f"\n✓ Indexed {final_count} headings in {elapsed:.1f}s  "
        f"(model={settings.embedding_model}, dim={settings.embedding_dimensions})"
    )

    if final_count != len(heading_infos):
        print(
            f"WARNING: expected {len(heading_infos)} but got {final_count}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
