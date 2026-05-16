"""Append-only writer for log.md — the ingestion changelog.

Idempotent: checks for an existing entry with the same file_id before appending.
Implemented in LW-2.
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

    Args:
        log_path: Path to the log.md file.
        file_id: UUID of the processed file.
        original_name: Original filename as uploaded by the user.
        created_pages: Slugs of newly created wiki pages.
        updated_pages: Slugs of updated wiki pages.
        cost_usd: Total LLM cost for this ingestion.
    """
    raise NotImplementedError("Implemented in LW-2")


def _entry_exists(log_path: Path, file_id: str) -> bool:
    """Return True if log.md already contains an entry for *file_id*.

    Args:
        log_path: Path to the log.md file.
        file_id: UUID to search for.
    """
    if not log_path.exists():
        return False
    return file_id in log_path.read_text(encoding="utf-8")
