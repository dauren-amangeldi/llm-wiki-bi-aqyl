"""Ingestion pipeline state machine.

State transitions:
    RECEIVED → STORED → SEARCHED → WRITTEN → LOGGED → DONE
                                ↘ FAILED (with step + error)

Each step is idempotent: safe to re-run with the same file_id.
Implemented in LW-9.
"""

from enum import StrEnum


class FileState(StrEnum):
    """All possible states of a file through the ingestion pipeline."""

    RECEIVED = "RECEIVED"
    STORED = "STORED"
    SEARCHED = "SEARCHED"
    WRITTEN = "WRITTEN"
    LOGGED = "LOGGED"
    DONE = "DONE"
    FAILED = "FAILED"


async def process_file(file_id: str) -> None:
    """Run the full ingestion pipeline for *file_id*.

    Steps:
        1. STORED  — parse the raw file
        2. SEARCHED — Search Agent finds relevant wiki pages
        3. WRITTEN  — Writer Agent creates/updates pages + index.md
        4. LOGGED   — append entry to log.md
        5. DONE     — mark complete in DB

    Each step checks idempotency before executing.
    Raises on unrecoverable failure after 3 retries (handled by Celery).

    Args:
        file_id: UUID of the file to process.
    """
    raise NotImplementedError("Implemented in LW-9")
