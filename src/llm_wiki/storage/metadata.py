"""SQLite metadata store — file_id, processing states, and timestamps.

Uses SQLAlchemy async ORM. Schema migrated via Alembic in Sprint 2+.
CRUD functions are implemented in LW-2; state machine integration in LW-9.

Inline migrations (pre-Alembic)
---------------------------------
Until Alembic is introduced (Sprint 2+), backward-compatible column additions
are handled by ``run_schema_migrations()``, called from ``main.py`` lifespan.
Each migration is **idempotent** — safe to run every startup.
"""

from datetime import datetime, timezone

import structlog
from sqlalchemy import JSON, DateTime, String, select, text, update as sa_update
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

logger = structlog.get_logger(__name__)


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all metadata models."""


class FileRecord(Base):
    """Metadata for a single uploaded file."""

    __tablename__ = "files"

    file_id: Mapped[str] = mapped_column(String, primary_key=True)
    # SHA-256 hex digest of the raw file content (nullable for legacy rows
    # created before LW-12.1; new uploads always populate this column).
    content_sha256: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    original_name: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="RECEIVED")
    state_history: Mapped[list[dict[str, str]]] = mapped_column(JSON, default=list)
    created_pages: Mapped[list[str]] = mapped_column(JSON, default=list)
    updated_pages: Mapped[list[str]] = mapped_column(JSON, default=list)
    cost_usd: Mapped[float | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# CRUD helpers — thin async wrappers used by the orchestrator (LW-9)
# ---------------------------------------------------------------------------


async def get_by_sha256(
    session: AsyncSession,
    sha256: str,
) -> FileRecord | None:
    """Return the most recent non-FAILED FileRecord with *sha256*, or None.

    Duplicate detection logic:
    - If a record exists with status != ``"FAILED"`` → caller should treat this
      as a duplicate and return the existing file_id without re-running the pipeline.
    - If only FAILED records exist for this hash → the previous run errored out;
      allow a fresh upload so the user can retry after fixing the underlying issue.

    Args:
        session: Active async SQLAlchemy session.
        sha256: SHA-256 hex digest (64 characters) of the file content.

    Returns:
        The matching FileRecord, or None.
    """
    result = await session.execute(
        select(FileRecord)
        .where(
            FileRecord.content_sha256 == sha256,
            FileRecord.status != "FAILED",  # allow re-upload after failure
        )
        .order_by(FileRecord.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def create_file_record(
    session: AsyncSession,
    file_id: str,
    original_name: str,
    content_sha256: str | None = None,
) -> FileRecord:
    """Insert a new FileRecord in RECEIVED state and return it.

    Args:
        session: Active async SQLAlchemy session.
        file_id: UUID7 identifier for the file.
        original_name: Original filename as provided by the uploader.

    Returns:
        The newly created and refreshed FileRecord.
    """
    record = FileRecord(
        file_id=file_id,
        content_sha256=content_sha256,
        original_name=original_name,
        status="RECEIVED",
        state_history=[],
        created_pages=[],
        updated_pages=[],
    )
    session.add(record)
    await session.commit()
    await session.refresh(record)
    return record


async def get_file_record(
    session: AsyncSession,
    file_id: str,
) -> FileRecord | None:
    """Fetch a FileRecord by file_id, or None if not found.

    Args:
        session: Active async SQLAlchemy session.
        file_id: UUID of the file to look up.

    Returns:
        The FileRecord, or None.
    """
    result = await session.execute(
        select(FileRecord).where(FileRecord.file_id == file_id)
    )
    return result.scalar_one_or_none()


async def update_file_status(
    session: AsyncSession,
    file_id: str,
    new_status: str,
) -> None:
    """Update the status field of an existing FileRecord.

    Args:
        session: Active async SQLAlchemy session.
        file_id: UUID of the file to update.
        new_status: New status string (e.g. ``"DONE"``, ``"FAILED"``).
    """
    await session.execute(
        sa_update(FileRecord)
        .where(FileRecord.file_id == file_id)
        .values(status=new_status, updated_at=datetime.now(timezone.utc))
    )
    await session.commit()


async def append_state_history(
    session: AsyncSession,
    file_id: str,
    state: str,
) -> None:
    """Append a state-transition entry to the file's state_history JSON list.

    Each entry has the shape ``{"state": "...", "at": "<iso8601>"}``.
    If the record does not exist this is a silent no-op.

    Args:
        session: Active async SQLAlchemy session.
        file_id: UUID of the file to update.
        state: State name to record (e.g. ``"STORED"``, ``"SEARCHED"``).
    """
    record = await get_file_record(session, file_id)
    if record is None:
        return
    history = list(record.state_history or [])
    history.append({"state": state, "at": datetime.now(timezone.utc).isoformat()})
    await session.execute(
        sa_update(FileRecord)
        .where(FileRecord.file_id == file_id)
        .values(state_history=history, updated_at=datetime.now(timezone.utc))
    )
    await session.commit()


# ---------------------------------------------------------------------------
# Inline schema migrations (pre-Alembic, Sprint 1)
# Called once from main.py lifespan — idempotent, safe every startup.
# ---------------------------------------------------------------------------

_MIGRATIONS: list[tuple[str, str, str]] = [
    # (migration_id, check_column, ddl)
    # Each tuple: human description, column name to probe, DDL to run if absent.
    (
        "LW-12.1 add content_sha256",
        "content_sha256",
        "ALTER TABLE files ADD COLUMN content_sha256 VARCHAR(64)",
    ),
]

_INDEX_DDL = (
    "CREATE INDEX IF NOT EXISTS ix_files_content_sha256 ON files (content_sha256)"
)


async def run_schema_migrations(conn: AsyncConnection) -> None:
    """Apply backward-compatible DDL changes to the *files* table.

    Each migration is idempotent: it checks whether the target column exists
    via ``PRAGMA table_info(files)`` before running ``ALTER TABLE``.
    Safe to call on every application startup — exits immediately when the
    schema is already up to date.

    Args:
        conn: Active async SQLAlchemy connection (not a session).  Typically
            obtained from ``engine.begin()`` in the lifespan hook.
    """
    # Fetch existing columns once
    result = await conn.execute(text("PRAGMA table_info(files)"))
    existing_cols = {row[1] for row in result.fetchall()}  # row[1] = column name

    for migration_id, column_name, ddl in _MIGRATIONS:
        if column_name not in existing_cols:
            logger.info("schema_migration_applying", migration=migration_id)
            await conn.execute(text(ddl))
            logger.info("schema_migration_done", migration=migration_id)
        else:
            logger.debug("schema_migration_skipped", migration=migration_id)

    # Always ensure the index exists (CREATE INDEX IF NOT EXISTS is idempotent)
    if "content_sha256" in existing_cols or any(
        col == "content_sha256" for _, col, _ in _MIGRATIONS
    ):
        await conn.execute(text(_INDEX_DDL))
