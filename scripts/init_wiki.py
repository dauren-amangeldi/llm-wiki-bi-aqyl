"""Initialize the data/ directory structure from scratch.

Run inside the container:
    docker compose exec api uv run python scripts/init_wiki.py
"""

from pathlib import Path

from llm_wiki.config import settings
from llm_wiki.storage.filesystem import ensure_dirs


def main() -> None:
    """Create all required data directories and seed empty marker files."""
    dirs = [settings.raw_dir, settings.wiki_dir, settings.chroma_dir]
    ensure_dirs(*dirs)

    for path, content in [
        (settings.index_path, "# Wiki Index\n"),
        (settings.log_path, "# Ingestion Log\n"),
        (settings.issues_path, "# Lint Agent Issues\n"),
        (settings.usage_log_path, ""),
    ]:
        if not path.exists():
            path.write_text(content, encoding="utf-8")
            print(f"Created {path}")
        else:
            print(f"Already exists: {path}")

    print("Done. data/ structure is ready.")


if __name__ == "__main__":
    main()
