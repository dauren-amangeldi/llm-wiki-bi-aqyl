"""SQLite metadata store — file_id, processing states, and timestamps.

Uses SQLAlchemy async ORM. Schema migrated via Alembic in Sprint 2+.
Implemented in LW-2 (basic CRUD) and LW-9 (state machine integration).
"""

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all metadata models."""


class FileRecord(Base):
    """Metadata for a single uploaded file."""

    __tablename__ = "files"

    file_id: Mapped[str] = mapped_column(String, primary_key=True)
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
