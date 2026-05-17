"""Append-only writer for log.md — the ingestion changelog.

Idempotent: checks for an existing entry with the same file_id before appending.
Reads the full file, appends the new entry, and writes back atomically.
"""

from datetime import datetime, timezone
from pathlib import Path

from llm_wiki.storage.filesystem import atomic_write


def append_log_entry(
    log_path: Path,
    file_id: str,
    original_name: str,
    created_pages: list[str],
    updated_pages: list[str],
    cost_usd: float,
) -> None:
    """Append a single ingestion record to log.md.

    Skips silently if an entry for *file_id* already exists (idempotency).
    Reads the current log, appends a formatted Markdown block, and writes
    back atomically via ``os.replace()``.

    Args:
        log_path: Path to the log.md file.
        file_id: UUID of the processed file.
        original_name: Original filename as uploaded by the user.
        created_pages: Slugs of newly created wiki pages.
        updated_pages: Slugs of updated wiki pages.
        cost_usd: Total LLM cost for this ingestion in USD.
    """
    if _entry_exists(log_path, file_id):
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    created_str = ", ".join(created_pages) if created_pages else "none"
    updated_str = ", ".join(updated_pages) if updated_pages else "none"

    entry = (
        f"\n## {timestamp} — {original_name}\n\n"
        f"- **File ID**: {file_id}\n"
        f"- **Created**: {created_str}\n"
        f"- **Updated**: {updated_str}\n"
        f"- **Cost**: ${cost_usd:.4f}\n"
    )

    current = log_path.read_text(encoding="utf-8") if log_path.exists() else "# Ingestion Log\n"
    if not current.endswith("\n"):
        current += "\n"
    atomic_write(log_path, current + entry)


def _entry_exists(log_path: Path, file_id: str) -> bool:
    """Return True if log.md already contains an entry for *file_id*.

    Args:
        log_path: Path to the log.md file.
        file_id: UUID to search for.
    """
    if not log_path.exists():
        return False
    return file_id in log_path.read_text(encoding="utf-8")
