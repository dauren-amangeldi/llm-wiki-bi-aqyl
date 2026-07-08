"""Initialize the data/ directory structure from scratch.

Run inside the container:
    docker compose exec api uv run python scripts/init_wiki.py
"""

from pathlib import Path

from llm_wiki.config import settings
from llm_wiki.storage.filesystem import ensure_dirs


def main() -> None:
    """Create the local data directories used by the local object-store backend.

    The wiki index, ingestion log and quality issues now live in Postgres
    (tables created on app startup), so no markdown marker files are needed.
    """
    ensure_dirs(settings.raw_dir, settings.wiki_dir)

    if not settings.usage_log_path.exists():
        settings.usage_log_path.write_text("", encoding="utf-8")
        print(f"Created {settings.usage_log_path}")

    print("Done. data/ structure is ready.")


if __name__ == "__main__":
    main()
