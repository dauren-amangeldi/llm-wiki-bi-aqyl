"""Rebuild the ChromaDB index from all pages currently in data/wiki/.

Run inside the container after restoring a backup or after switching models:
    docker compose exec api uv run python scripts/reindex.py

Implemented in LW-11.
"""


def main() -> None:
    """Scan data/wiki/*.md and re-embed all headings into ChromaDB."""
    raise NotImplementedError("Implemented in LW-11")


if __name__ == "__main__":
    main()
