"""Rebuild the ChromaDB heading index from the current data/index.md.

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
        description="Rebuild ChromaDB heading index from index.md",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview headings without writing to ChromaDB.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Override EMBEDDING_BATCH_SIZE from config (0 = use config value).",
    )
    return parser.parse_args()


def main() -> None:
    """Scan index.md and re-embed all headings into ChromaDB."""
    args = _parse_args()

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
    store = EmbeddingStore(chroma_path=settings.chroma_dir, llm_client=llm)

    print(f"Clearing existing collection …")
    store.clear()

    print(f"Embedding {len(heading_infos)} headings (model={settings.embedding_model}) …")
    t0 = time.perf_counter()

    try:
        store.upsert_many(heading_infos)
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
