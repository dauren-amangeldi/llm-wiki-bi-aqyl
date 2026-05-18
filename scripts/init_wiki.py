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

    lang = settings.wiki_language.lower()
    index_header = "# Карта знаний\n" if lang == "ru" else "# Wiki Index\n"
    log_header = "# Журнал изменений\n" if lang == "ru" else "# Ingestion Log\n"
    issues_header = "# Отчёты проверок\n" if lang == "ru" else "# Lint Agent Issues\n"

    for path, content in [
        (settings.index_path, index_header),
        (settings.log_path, log_header),
        (settings.issues_path, issues_header),
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
