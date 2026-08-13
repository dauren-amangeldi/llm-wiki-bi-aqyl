"""PostgreSQL metadata store — files, cases, chat history, users, skills.

Uses SQLAlchemy async ORM (psycopg driver). Tables are created on startup via
``Base.metadata.create_all``; schema changes should move to Alembic when needed.
"""

from dataclasses import dataclass
from datetime import datetime, timezone

import structlog
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Index,
    Integer,
    String,
    Text,
    cast,
    create_engine,
    select,
    text,
    update as sa_update,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from llm_wiki.config import settings


logger = structlog.get_logger(__name__)

_DEV_USER_ID = "dev-user"
_DEV_USER_NAME = "Dev User"
_DEV_USER_ROLE = "admin"


# ---------------------------------------------------------------------------
# Idempotent column back-fill migrations
# ---------------------------------------------------------------------------
# ``Base.metadata.create_all`` creates missing *tables* but never adds *columns*
# to a table that already exists. These "ADD COLUMN IF NOT EXISTS" statements
# backfill columns that later features added to pre-existing tables — the
# failure reason on ``files`` and the owner-scoped private flags on
# ``files`` / ``chunk_embeddings`` / ``cases`` — so an older database
# self-migrates on startup instead of needing a manual DBA step. Idempotent and
# safe to re-run; mirrors the pattern in ``wiki_fts.ensure_wiki_fts_table``.
# Index names match SQLAlchemy's default (``ix_<table>_<column>``) so a fresh
# ``create_all`` and this backfill never create a duplicate.
_COLUMN_MIGRATIONS: tuple[str, ...] = (
    # Trigram similarity for fuzzy (typo-tolerant) search over titles/names.
    "CREATE EXTENSION IF NOT EXISTS pg_trgm",
    "ALTER TABLE files ADD COLUMN IF NOT EXISTS error text",
    "ALTER TABLE files ADD COLUMN IF NOT EXISTS sensitive boolean NOT NULL DEFAULT false",
    "ALTER TABLE files ADD COLUMN IF NOT EXISTS owner varchar",
    "CREATE INDEX IF NOT EXISTS ix_files_owner ON files (owner)",
    "ALTER TABLE chunk_embeddings ADD COLUMN IF NOT EXISTS sensitive boolean NOT NULL DEFAULT false",
    "ALTER TABLE chunk_embeddings ADD COLUMN IF NOT EXISTS owner varchar",
    "CREATE INDEX IF NOT EXISTS ix_chunk_embeddings_owner ON chunk_embeddings (owner)",
    "ALTER TABLE cases ADD COLUMN IF NOT EXISTS sensitive boolean NOT NULL DEFAULT false",
    "ALTER TABLE cases ADD COLUMN IF NOT EXISTS owner varchar",
    "ALTER TABLE cases ADD COLUMN IF NOT EXISTS scope varchar NOT NULL DEFAULT 'internal'",
    # One row per (document, kind) is the store's contract — enforce it so two
    # concurrent generations can't insert duplicates (check-then-insert race
    # became real once artifact workers run in parallel). Dedupe first: keep an
    # arbitrary survivor (dupes are already an anomaly), then add the index.
    "DELETE FROM artifacts a USING artifacts b"
    " WHERE a.document_id = b.document_id AND a.kind = b.kind AND a.ctid < b.ctid",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_artifacts_document_kind"
    " ON artifacts (document_id, kind)",
    # Ops monitoring: who requested a generation and when it actually ran.
    "ALTER TABLE artifacts ADD COLUMN IF NOT EXISTS requested_by varchar",
    "ALTER TABLE artifacts ADD COLUMN IF NOT EXISTS started_at timestamptz",
    "ALTER TABLE artifacts ADD COLUMN IF NOT EXISTS finished_at timestamptz",
    "CREATE INDEX IF NOT EXISTS ix_cases_owner ON cases (owner)",
    "ALTER TABLE cases ADD COLUMN IF NOT EXISTS tags json",
    "ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS citation_quotes json",
    "ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS citation_cases json",
    "ALTER TABLE artifacts ADD COLUMN IF NOT EXISTS error text",
    "ALTER TABLE twin_personas ADD COLUMN IF NOT EXISTS color varchar NOT NULL DEFAULT ''",
    "ALTER TABLE twin_personas ADD COLUMN IF NOT EXISTS description varchar NOT NULL DEFAULT ''",
)


async def ensure_column_migrations(conn: AsyncConnection) -> None:
    """Backfill columns added to pre-existing tables (idempotent).

    ``create_all`` does not ALTER existing tables, so these run on startup in
    the same transaction as ``create_all``. A no-op once the columns exist.
    """
    for stmt in _COLUMN_MIGRATIONS:
        await conn.execute(text(stmt))


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all metadata models."""


# ---------------------------------------------------------------------------
# Vector search (pgvector) — replaces the former ChromaDB collections.
# ---------------------------------------------------------------------------

_EMBED_DIM = settings.embedding_dimensions


class HeadingEmbedding(Base):
    """One embedding per wiki page heading (title). Former ``headings`` collection."""

    __tablename__ = "heading_embeddings"

    slug: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str] = mapped_column(String, nullable=False, default="")
    section: Mapped[str] = mapped_column(String, nullable=False, default="")
    level: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    file_id: Mapped[str] = mapped_column(String, nullable=False, default="", index=True)
    last_indexed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    embedding: Mapped[list[float]] = mapped_column(Vector(_EMBED_DIM), nullable=False)

    __table_args__ = (
        Index(
            "ix_heading_embeddings_vec",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class ChunkEmbedding(Base):
    """One embedding per ~500-token page-body chunk. Former ``chunks`` collection."""

    __tablename__ = "chunk_embeddings"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # f"{slug}#{idx:04d}"
    slug: Mapped[str] = mapped_column(String, nullable=False, index=True)
    title: Mapped[str] = mapped_column(String, nullable=False, default="")
    section: Mapped[str] = mapped_column(String, nullable=False, default="")
    chunk_idx: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    file_id: Mapped[str] = mapped_column(String, nullable=False, default="", index=True)
    # Access control: sensitive chunks are only returned to their owner (Q&A).
    # Non-sensitive chunks have sensitive=False / owner=NULL (visible to all).
    sensitive: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    owner: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    document: Mapped[str] = mapped_column(Text, nullable=False, default="")
    embedding: Mapped[list[float]] = mapped_column(Vector(_EMBED_DIM), nullable=False)

    __table_args__ = (
        Index(
            "ix_chunk_embeddings_vec",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class EmbeddingMeta(Base):
    """Records the model + dim each vector table was built with (mismatch guard)."""

    __tablename__ = "embedding_meta"

    scope: Mapped[str] = mapped_column(String, primary_key=True)  # "headings" | "chunks"
    model: Mapped[str] = mapped_column(String, nullable=False)
    dim: Mapped[int] = mapped_column(Integer, nullable=False)


class WikiIndexEntry(Base):
    """The wiki knowledge-map index — one row per page (replaces index.md)."""

    __tablename__ = "wiki_index"

    slug: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str] = mapped_column(String, nullable=False, default="")
    section: Mapped[str] = mapped_column(String, nullable=False, default="General")
    level: Mapped[int] = mapped_column(Integer, nullable=False, default=2)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class IssuesReport(Base):
    """Rendered quality-issues sections (replaces issues.md). One row per section."""

    __tablename__ = "issues_report"

    section: Mapped[str] = mapped_column(String, primary_key=True)  # auto-detected | llm-flagged
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


# Synchronous engine for the (synchronous) vector stores. The app's main engine
# is async; embedding upsert/query are sync (LLMClient.embed is sync), so they
# share their own sync psycopg engine against the same DATABASE_URL.
_sync_engine: Engine | None = None


def get_sync_engine() -> Engine:
    """Return a lazily-created synchronous engine to the configured database."""
    global _sync_engine
    if _sync_engine is None:
        _sync_engine = create_engine(settings.database_url, pool_pre_ping=True)
    return _sync_engine


class User(Base):
    """Minimal user record for dev-user identity (get_current_user)."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False, default="employee")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


class AllowedUser(Base):
    """Access-control list for Keycloak-authenticated callers (LW-auth).

    When ``AUTH_ENABLED`` is on, a verified Keycloak identity (``isBIGroupPerson``)
    must additionally have a matching, non-``blocked`` row here to use the API —
    a strict whitelist. ``is_admin`` grants the admin role (source of truth for
    roles, independent of Keycloak realm roles).

    The gate works in both directions: presence (with ``blocked=False``) allows;
    ``blocked=True`` denies even a whitelisted account. Blocking is a rare, one-off
    action — flip ``blocked`` (via the ``manage_access`` script or a data
    migration) and keep the row for an audit trail rather than deleting it.
    Emails are stored lower-cased.
    """

    __tablename__ = "allowed_users"

    email: Mapped[str] = mapped_column(String, primary_key=True)
    is_admin: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    blocked: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    note: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


@dataclass(frozen=True)
class AccessDecision:
    """Outcome of the whitelist check for a verified identity."""

    allowed: bool
    is_admin: bool
    reason: str  # "ok" | "not_whitelisted" | "blocked"


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
    # Date-partitioned object-store key (YYYY/MM/DD/<file_id><ext>). NULL for
    # rows created before date-partitioning (read via the legacy raw/ path).
    raw_key: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="RECEIVED")
    # Human-readable failure reason, set when status becomes FAILED. Surfaced in
    # GET /files/{id} and the status stream so the cause is visible without logs.
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Access control (sensitive files): indexed but owner-scoped — excluded from
    # the shared wiki/search and only retrievable by their owner.
    sensitive: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    owner: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
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


class CaseRecord(Base):
    """A user-created case (topic container) grouping related documents."""

    __tablename__ = "cases"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    doc_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    # Fixed-taxonomy tags (see llm_wiki.taxonomy): auto-assigned at creation,
    # then user-editable. Used to filter cases in the navigation.
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)
    # Case-level privacy: a private (sensitive) case keeps its files owner-scoped
    # — indexed but never surfaced in the shared wiki/search for other users.
    sensitive: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    owner: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    # Источник опыта: internal (внутренний опыт BI) | external (мировой опыт).
    # Выбирается автором при создании кейса; двигает фильтр на главной.
    scope: Mapped[str] = mapped_column(String, nullable=False, default="internal")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class ArtifactRecord(Base):
    """A generated studio artifact for a document/case — test, report,
    presentation, card, or infographic — with per-language versions.

    One row per (document_id, kind): regenerating updates the row and appends
    or replaces the language version. Content shape is kind-specific and matches
    the frontend renderers (e.g. test → {questions:[...]}, report →
    {executive_summary, metrics, sections}).
    """

    __tablename__ = "artifacts"

    artifact_id: Mapped[str] = mapped_column(String, primary_key=True)
    document_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    kind: Mapped[str] = mapped_column(String, nullable=False, index=True)
    # versions: [{"language": "ru", "content": {...}}, ...]
    versions: Mapped[list[dict]] = mapped_column(JSON, default=list)
    # "pending" while a background task generates it, "ready" when done, "failed"
    # on error (async generation — heavy artifacts run in a Celery worker).
    status: Mapped[str] = mapped_column(String, nullable=False, default="ready")
    # Human-readable failure reason when status == "failed".
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    # Ops monitoring (генерации в Grafana): кто запросил и когда реально
    # началась/закончилась работа воркера. NULL для legacy-строк.
    requested_by: Mapped[str | None] = mapped_column(String, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ChatRecord(Base):
    """A single persisted chat turn, scoped to a user + (document|case).

    Powers chat history: a user's questions and the assistant's answers are
    stored so the conversation can be reloaded and continued across sessions.
    """

    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_key: Mapped[str] = mapped_column(String, nullable=False, index=True)
    scope_type: Mapped[str] = mapped_column(String, nullable=False)  # "document" | "case"
    scope_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    role: Mapped[str] = mapped_column(String, nullable=False)  # "user" | "assistant"
    text: Mapped[str] = mapped_column(String, nullable=False)
    citations: Mapped[list[str]] = mapped_column(JSON, default=list)
    # anchor → short supporting quote, so reloaded history keeps the [n] hover
    # card + reader highlight (only anchors were persisted before).
    citation_quotes: Mapped[dict[str, str]] = mapped_column(JSON, default=dict)
    # anchor → {"id", "title"} of the case the cited source belongs to, so the
    # reloaded answer keeps its "source case" chip next to each citation.
    citation_cases: Mapped[dict[str, dict]] = mapped_column(JSON, default=dict)
    model_name: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )


class Skill(Base):
    """Role-based system prompt for AI modes and positions (LW-N11)."""

    __tablename__ = "skills"

    role: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(String, nullable=False, default="")
    system_prompt: Mapped[str] = mapped_column(String, nullable=False, default="")
    active: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


# Roles grouped for frontend slug mapping (modes/* vs positions/*)
_MODE_ROLES: frozenset[str] = frozenset({"advisor", "expert", "library"})

_DEFAULT_SKILL_SEEDS: list[dict[str, str | int]] = [
    {
        "role": "advisor",
        "title": "AI-Советник",
        "description": "Задаёт вектор решения, предлагает стратегию на основе кейсов",
        "system_prompt": (
            "Ты AI-советник BI Group. Анализируй кейсы из базы знаний и предлагай "
            "конкретные стратегические действия, привязанные к реальным case_id. "
            "Отвечай строго на языке запроса. Возвращай только JSON по схеме."
        ),
    },
    {
        "role": "expert",
        "title": "Эксперт",
        "description": "Глубокий анализ кейса с детальными рекомендациями",
        "system_prompt": (
            "Ты эксперт BI Group. Давай глубокий анализ кейсов с детальными "
            "рекомендациями и ссылками на источники. Отвечай строго на языке запроса."
        ),
    },
    {
        "role": "library",
        "title": "Библиотека",
        "description": "Поиск по всей базе знаний с цитатами из источников",
        "system_prompt": (
            "Ты библиотечный ассистент BI Group. Находи релевантные материалы "
            "и цитируй источники дословно. Не выдумывай факты."
        ),
    },
    {
        "role": "employee",
        "title": "Сотрудник",
        "description": "Базовый режим для линейных сотрудников",
        "system_prompt": (
            "Ты помощник для линейного сотрудника BI Group. Объясняй простым языком, "
            "фокусируйся на практическом применении кейсов в ежедневной работе."
        ),
    },
    {
        "role": "finance",
        "title": "Финансы",
        "description": "Финансовый анализ, бюджетирование, отчётность",
        "system_prompt": (
            "Ты финансовый аналитик BI Group. Акцентируй бюджет, ROI, "
            "финансовые риски и метрики из кейсов. Не выдумывай цифры."
        ),
    },
    {
        "role": "gd",
        "title": "Генеральный директор",
        "description": "Стратегические решения для топ-менеджмента",
        "system_prompt": (
            "Ты стратегический советник для топ-менеджмента BI Group. "
            "Фокусируйся на стратегических выводах, рисках и возможностях масштабирования."
        ),
    },
    {
        "role": "hr",
        "title": "HR",
        "description": "Управление персоналом, подбор, развитие команды",
        "system_prompt": (
            "Ты HR-эксперт BI Group. Акцентируй развитие персонала, "
            "компетенции, адаптацию и практики из кейсов."
        ),
    },
    {
        "role": "legal",
        "title": "Юридический",
        "description": "Юридическая экспертиза, комплаенс, договоры",
        "system_prompt": (
            "Ты юридический советник BI Group. Выделяй правовые риски, "
            "комплаенс и договорные практики из кейсов. Не давай юридических заключений вне источников."
        ),
    },
    {
        "role": "pm",
        "title": "Project Manager",
        "description": "Управление проектами, планирование, контроль",
        "system_prompt": (
            "Ты project manager BI Group. Фокусируй сроки, зависимости, "
            "риски и lean-практики из кейсов. Предлагай конкретные шаги."
        ),
    },
    {
        "role": "pto",
        "title": "PTO / Инженерия",
        "description": "Техническая экспертиза, инженерные решения",
        "system_prompt": (
            "Ты инженерный эксперт BI Group. Акцентируй технические решения, "
            "BIM, контроль качества и строительные практики из кейсов."
        ),
    },
]


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
    raw_key: str | None = None,
    sensitive: bool = False,
    owner: str | None = None,
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
        raw_key=raw_key,
        status="RECEIVED",
        sensitive=sensitive,
        owner=owner,
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
    error: str | None = None,
) -> None:
    """Update the status (and optionally the failure reason) of a FileRecord.

    Args:
        session: Active async SQLAlchemy session.
        file_id: UUID of the file to update.
        new_status: New status string (e.g. ``"DONE"``, ``"FAILED"``).
        error: Failure reason to persist (truncated); ignored when None.
    """
    values: dict[str, object] = {
        "status": new_status,
        "updated_at": datetime.now(timezone.utc),
    }
    if error is not None:
        values["error"] = error[:2000]
    await session.execute(
        sa_update(FileRecord).where(FileRecord.file_id == file_id).values(**values)
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
# User CRUD (LW-N1)
# ---------------------------------------------------------------------------


async def get_user_by_id(session: AsyncSession, user_id: str) -> User | None:
    """Return a User by primary key, or None if not found."""
    result = await session.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()


async def get_or_create_user(
    session: AsyncSession,
    user_id: str,
    name: str,
    role: str,
) -> User:
    """Return an existing user or insert a new one (idempotent on *user_id*)."""
    existing = await get_user_by_id(session, user_id)
    if existing is not None:
        return existing
    user = User(id=user_id, name=name, role=role)
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


async def ensure_dev_user(session: AsyncSession) -> User:
    """Return the default development user, creating it if absent."""
    return await get_or_create_user(
        session, _DEV_USER_ID, _DEV_USER_NAME, _DEV_USER_ROLE
    )


# ---------------------------------------------------------------------------
# Access control — Keycloak whitelist + admin roles (LW-auth)
# ---------------------------------------------------------------------------

# Seed for a fresh database. The demo account is whitelisted + admin so the app
# is usable out of the box; its Keycloak password is set in Keycloak, not here.
_DEFAULT_ALLOWED_SEEDS: list[dict[str, object]] = [
    {"email": "demo@bi.group", "is_admin": True, "note": "seeded demo account"},
]


def _norm_email(email: str) -> str:
    """Normalise an email for stable lookups (trim + lowercase)."""
    return email.strip().lower()


async def get_allowed_user(session: AsyncSession, email: str) -> AllowedUser | None:
    """Return the access-list row for *email* (case-insensitive), or None."""
    result = await session.execute(
        select(AllowedUser).where(AllowedUser.email == _norm_email(email))
    )
    return result.scalar_one_or_none()


async def access_for_email(session: AsyncSession, email: str) -> AccessDecision:
    """Decide whether *email* may use the API and whether it is an admin.

    Default (OPEN, Zebo-style): any authenticated user is allowed; the
    ``allowed_users`` row, when present, only grants the admin role — its
    absence no longer denies access.

    Strict mode (``settings.auth_strict_allowlist``): deny-by-default — the
    email must have a non-blocked ``allowed_users`` row; unknown → denied,
    ``blocked`` row → denied.
    """
    from llm_wiki.config import settings

    row = await get_allowed_user(session, email)
    if settings.auth_strict_allowlist:
        if row is None:
            return AccessDecision(False, False, "not_whitelisted")
        if row.blocked:
            return AccessDecision(False, False, "blocked")
        return AccessDecision(True, bool(row.is_admin), "ok")
    # Open access: everyone in; a non-blocked admin row still grants admin.
    return AccessDecision(True, bool(row and row.is_admin and not row.blocked), "ok")


async def allowed_users_count(session: AsyncSession) -> int:
    """Return the number of rows in the access-list table."""
    result = await session.execute(select(AllowedUser))
    return len(result.scalars().all())


async def seed_allowed_users(session: AsyncSession) -> int:
    """Insert the default access-list rows when absent (idempotent).

    Returns the number of rows inserted.
    """
    inserted = 0
    for row in _DEFAULT_ALLOWED_SEEDS:
        email = _norm_email(str(row["email"]))
        if await get_allowed_user(session, email) is not None:
            continue
        note = str(row.get("note") or "") or None
        session.add(
            AllowedUser(email=email, is_admin=bool(row.get("is_admin", False)), note=note)
        )
        inserted += 1
    if inserted:
        await session.commit()
    return inserted


# ---------------------------------------------------------------------------
# Slug ↔ file_id mapping (LW-N3 backfill)
# ---------------------------------------------------------------------------


async def build_slug_to_file_id_map(session: AsyncSession) -> dict[str, str]:
    """Map wiki slugs to the most recent source *file_id* (LW-N3 backfill)."""
    result = await session.execute(select(FileRecord).order_by(FileRecord.created_at))
    records = result.scalars().all()
    mapping: dict[str, str] = {}
    for record in records:
        for slug in list(record.created_pages or []) + list(record.updated_pages or []):
            mapping[slug] = record.file_id
    return mapping


# ---------------------------------------------------------------------------
# Skills CRUD (LW-N11 / LW-N12)
# ---------------------------------------------------------------------------


def skill_role_to_slug(role: str) -> str:
    """Map a DB role key to the frontend slug (``modes/advisor``, etc.)."""
    if role in _MODE_ROLES:
        return f"modes/{role}"
    return f"positions/{role}"


def skill_slug_to_role(slug: str) -> str:
    """Parse a frontend slug back to the DB role key."""
    if slug.startswith("modes/"):
        return slug[len("modes/") :]
    if slug.startswith("positions/"):
        return slug[len("positions/") :]
    return slug


async def list_skills(session: AsyncSession) -> list[Skill]:
    """Return all skills ordered by role."""
    result = await session.execute(select(Skill).order_by(Skill.role))
    return list(result.scalars().all())


async def get_skill(session: AsyncSession, role: str) -> Skill | None:
    """Fetch a skill by role key, or None if not found."""
    result = await session.execute(select(Skill).where(Skill.role == role))
    return result.scalar_one_or_none()


async def update_skill(
    session: AsyncSession,
    role: str,
    *,
    system_prompt: str | None = None,
    active: int | None = None,
) -> Skill | None:
    """Update ``system_prompt`` and/or ``active`` for *role*."""
    skill = await get_skill(session, role)
    if skill is None:
        return None
    values: dict[str, object] = {"updated_at": datetime.now(timezone.utc)}
    if system_prompt is not None:
        values["system_prompt"] = system_prompt
    if active is not None:
        values["active"] = active
    await session.execute(sa_update(Skill).where(Skill.role == role).values(**values))
    await session.commit()
    return await get_skill(session, role)


async def skills_count(session: AsyncSession) -> int:
    """Return the number of rows in the skills table."""
    result = await session.execute(select(Skill))
    return len(result.scalars().all())


async def seed_skills(session: AsyncSession) -> int:
    """Insert default skills when absent (idempotent). Returns rows inserted."""
    inserted = 0
    for row in _DEFAULT_SKILL_SEEDS:
        role = str(row["role"])
        existing = await get_skill(session, role)
        if existing is not None:
            continue
        session.add(
            Skill(
                role=role,
                title=str(row["title"]),
                description=str(row["description"]),
                system_prompt=str(row["system_prompt"]),
                active=int(row.get("active", 1)),
            )
        )
        inserted += 1
    if inserted:
        await session.commit()
    return inserted


async def resolve_skill_system_prompt(session: AsyncSession, role: str) -> str | None:
    """Return the active system prompt for *role*, or None when missing/inactive."""
    skill = await get_skill(session, role)
    if skill is None or not skill.active:
        return None
    prompt = skill.system_prompt.strip()
    return prompt or None


# ---------------------------------------------------------------------------
# Chat history CRUD
# ---------------------------------------------------------------------------


async def append_chat_message(
    session: AsyncSession,
    *,
    user_key: str,
    scope_type: str,
    scope_id: str,
    role: str,
    text_body: str,
    citations: list[str] | None = None,
    citation_quotes: dict[str, str] | None = None,
    citation_cases: dict[str, dict] | None = None,
    model_name: str | None = None,
) -> ChatRecord:
    """Persist one chat turn and return the saved row."""
    record = ChatRecord(
        user_key=user_key,
        scope_type=scope_type,
        scope_id=scope_id,
        role=role,
        text=text_body,
        citations=citations or [],
        citation_quotes=citation_quotes or {},
        citation_cases=citation_cases or {},
        model_name=model_name,
    )
    session.add(record)
    await session.commit()
    return record


async def case_for_file(
    session: AsyncSession, file_id: str
) -> tuple[str, str] | None:
    """Return ``(case_id, case_title)`` of the case that owns *file_id*, if any.

    A source belongs to a case when its ``file_id`` is in the case's
    ``doc_ids``; if it's in several, the most recently created case wins. Used to
    label an answer's citations with their source case (the "source case" chip).
    JSONB containment keeps this a single indexed-friendly lookup instead of a
    full case scan.
    """
    if not file_id:
        return None
    stmt = (
        select(CaseRecord.id, CaseRecord.title)
        .where(cast(CaseRecord.doc_ids, JSONB).contains([file_id]))
        .order_by(CaseRecord.created_at.desc())
        .limit(1)
    )
    row = (await session.execute(stmt)).first()
    return (row[0], row[1]) if row else None


async def list_chat_messages(
    session: AsyncSession,
    *,
    user_key: str,
    scope_type: str,
    scope_id: str,
    limit: int = 200,
) -> list[ChatRecord]:
    """Return a user's chat turns for a scope, oldest-first."""
    stmt = (
        select(ChatRecord)
        .where(
            ChatRecord.user_key == user_key,
            ChatRecord.scope_type == scope_type,
            ChatRecord.scope_id == scope_id,
        )
        .order_by(ChatRecord.created_at, ChatRecord.id)
        .limit(limit)
    )
    return list((await session.execute(stmt)).scalars().all())


async def clear_chat_messages(
    session: AsyncSession,
    *,
    user_key: str,
    scope_type: str,
    scope_id: str,
) -> int:
    """Delete a user's chat turns for a scope. Returns count removed."""
    rows = await list_chat_messages(
        session, user_key=user_key, scope_type=scope_type, scope_id=scope_id, limit=10000
    )
    for row in rows:
        await session.delete(row)
    await session.commit()
    return len(rows)


# ---------------------------------------------------------------------------
# Twins council (BI-AQYL-TWINS) — personas, presets, sessions, messages
# Ported from feature/round2, reconciled to this schema (timezone.utc; wiki in
# Postgres; suggest degrades gracefully without case-similarity).
# ---------------------------------------------------------------------------


class TwinPersona(Base):
    """A Twins council persona. Content (name/lens/system_prompt/domain_weights)
    is seeded from editable .md files in ``personas/twins/`` — see
    ``seed_twin_personas``."""

    __tablename__ = "twin_personas"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    inspiration: Mapped[str] = mapped_column(String, nullable=False)
    real_name: Mapped[str] = mapped_column(String, nullable=False, default="")
    track: Mapped[str] = mapped_column(String, nullable=False)  # "tech" | "dev"
    pinned: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    lens: Mapped[str] = mapped_column(String, nullable=False)
    system_prompt: Mapped[str] = mapped_column(String, nullable=False)
    domain_weights: Mapped[dict[str, float]] = mapped_column(JSON, default=dict)
    avatar_init: Mapped[str] = mapped_column(String, nullable=False)
    # Persona accent hex (bubble border/tint, avatar ring, session-row accent).
    color: Mapped[str] = mapped_column(String, nullable=False, default="")
    # One-sentence philosophy line shown on the gallery card (mock design).
    description: Mapped[str] = mapped_column(String, nullable=False, default="")
    active: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class TwinPreset(Base):
    """A ready-made triad preset for the Twins council."""

    __tablename__ = "twin_presets"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    persona_ids: Mapped[list[str]] = mapped_column(JSON, default=list)


class TwinSession(Base):
    """A single Twins council run — which case, which personas."""

    __tablename__ = "twin_sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    case_id: Mapped[str] = mapped_column(String, nullable=False)
    persona_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    created_by: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    # Outcome journal: did the council's verdict hold up? "" = not reviewed yet.
    outcome: Mapped[str] = mapped_column(String, nullable=False, default="")  # "" | confirmed | refuted
    outcome_note: Mapped[str] = mapped_column(String(1000), nullable=False, default="")
    outcome_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class TwinMessage(Base):
    """A single message in a Twins chat transcript."""

    __tablename__ = "twin_messages"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    role: Mapped[str] = mapped_column(String, nullable=False)  # user | persona | verdict
    persona_id: Mapped[str | None] = mapped_column(String, nullable=True)
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


_DEFAULT_TWIN_PRESET_SEEDS: list[dict[str, object]] = [
    {"id": "preset-tech-transform", "name": "Технотрансформация", "persona_ids": ["musk", "huang", "nadella"]},
    {"id": "preset-should-build", "name": "Стоит ли строить", "persona_ids": ["buffett", "rakhimbayev", "musk"]},
    {"id": "preset-how-to-build", "name": "Каким строить", "persona_ids": ["rakhimbayev", "jobs", "huang"]},
    {"id": "preset-sell-adopt", "name": "Как продать / внедрить", "persona_ids": ["bezos", "altman", "nadella"]},
]


async def twin_personas_count(session: AsyncSession) -> int:
    """Number of persona rows (used to log seeding on first boot)."""
    result = await session.execute(select(TwinPersona))
    return len(result.scalars().all())


async def get_twin_persona(session: AsyncSession, persona_id: str) -> TwinPersona | None:
    return await session.get(TwinPersona, persona_id)


async def list_twin_personas(session: AsyncSession) -> list[TwinPersona]:
    """All active personas."""
    result = await session.execute(select(TwinPersona).where(TwinPersona.active == 1))
    return list(result.scalars().all())


async def list_twin_presets(session: AsyncSession) -> list[TwinPreset]:
    result = await session.execute(select(TwinPreset))
    return list(result.scalars().all())


async def seed_twin_personas(session: AsyncSession) -> int:
    """Upsert personas from .md files + insert-if-absent presets.

    Upsert (not insert-only) so editing a persona file and restarting applies
    the change without a migration. Returns the number of personas newly
    inserted (updates aren't counted).
    """
    from llm_wiki.storage.persona_files import load_persona_files

    inserted = 0
    seeded_ids: set[str] = set()
    for row in load_persona_files():
        persona_id = str(row["id"])
        seeded_ids.add(persona_id)
        pinned_int = int(bool(row["pinned"]))
        existing = await get_twin_persona(session, persona_id)
        if existing is None:
            session.add(
                TwinPersona(
                    id=persona_id,
                    name=str(row["name"]),
                    inspiration=str(row["inspiration"]),
                    real_name=str(row["real_name"]),
                    track=str(row["track"]),
                    pinned=pinned_int,
                    lens=str(row["lens"]),
                    system_prompt=str(row["system_prompt"]),
                    domain_weights=row["domain_weights"],  # type: ignore[arg-type]
                    avatar_init=str(row["avatar_init"]),
                    color=str(row.get("color", "")),
                    description=str(row.get("description", "")),
                )
            )
            inserted += 1
        else:
            existing.name = str(row["name"])
            existing.inspiration = str(row["inspiration"])
            existing.real_name = str(row["real_name"])
            existing.track = str(row["track"])
            existing.pinned = pinned_int
            existing.lens = str(row["lens"])
            existing.system_prompt = str(row["system_prompt"])
            existing.domain_weights = row["domain_weights"]  # type: ignore[assignment]
            existing.avatar_init = str(row["avatar_init"])
            existing.color = str(row.get("color", ""))
            existing.description = str(row.get("description", ""))
            existing.active = 1

    # A persona whose seed file was removed is retired: deactivate it so it
    # leaves the roster but stays resolvable for old sessions' transcripts.
    all_rows = await session.execute(select(TwinPersona))
    for persona in all_rows.scalars().all():
        if persona.id not in seeded_ids and persona.active:
            persona.active = 0

    # Upsert (was insert-if-absent): the default triads reference persona ids,
    # so retiring a persona must also refresh the stored preset line-ups.
    for preset_row in _DEFAULT_TWIN_PRESET_SEEDS:
        preset_id = str(preset_row["id"])
        existing_preset = await session.get(TwinPreset, preset_id)
        if existing_preset is not None:
            existing_preset.name = str(preset_row["name"])
            existing_preset.persona_ids = preset_row["persona_ids"]  # type: ignore[assignment]
            continue
        session.add(
            TwinPreset(
                id=preset_id,
                name=str(preset_row["name"]),
                persona_ids=preset_row["persona_ids"],  # type: ignore[arg-type]
            )
        )

    await session.commit()
    return inserted


async def create_twin_session(
    session: AsyncSession, *, case_id: str, persona_ids: list[str], created_by: str
) -> TwinSession:
    """Create and persist a new Twins council session."""
    now = datetime.now(timezone.utc)
    row = TwinSession(
        id=f"twin-session-{int(now.timestamp() * 1000):x}",
        case_id=case_id,
        persona_ids=persona_ids,
        created_by=created_by,
        created_at=now,
    )
    session.add(row)
    await session.commit()
    return row


async def append_twin_message(
    session: AsyncSession,
    *,
    session_id: str,
    role: str,
    persona_id: str | None,
    seq: int,
    content: dict[str, object],
) -> TwinMessage:
    """Persist one message immediately, so a dropped stream keeps partial history."""
    row = TwinMessage(
        id=f"twin-msg-{session_id}-{seq}",
        session_id=session_id,
        role=role,
        persona_id=persona_id,
        seq=seq,
        content=content,
        created_at=datetime.now(timezone.utc),
    )
    session.add(row)
    await session.commit()
    return row


async def get_twin_session_messages(session: AsyncSession, session_id: str) -> list[TwinMessage]:
    """All messages for a session, ordered by seq."""
    result = await session.execute(
        select(TwinMessage).where(TwinMessage.session_id == session_id).order_by(TwinMessage.seq)
    )
    return list(result.scalars().all())


async def list_twin_sessions(session: AsyncSession, case_id: str) -> list[TwinSession]:
    """Past councils for a case, newest first (outcome journal view)."""
    stmt = (
        select(TwinSession)
        .where(TwinSession.case_id == case_id)
        .order_by(TwinSession.created_at.desc())
    )
    return list((await session.execute(stmt)).scalars().all())


async def delete_twin_session(session: AsyncSession, session_id: str) -> bool:
    """Delete a council session and its whole transcript. False if unknown."""
    row = await session.get(TwinSession, session_id)
    if not row:
        return False
    await session.execute(
        TwinMessage.__table__.delete().where(TwinMessage.session_id == session_id)
    )
    await session.delete(row)
    await session.commit()
    return True


async def update_twin_session_personas(
    session: AsyncSession, session_id: str, persona_ids: list[str]
) -> bool:
    """Change a session's council line-up («Изменить состав»). False if unknown."""
    row = await session.get(TwinSession, session_id)
    if not row:
        return False
    row.persona_ids = persona_ids
    await session.commit()
    return True


async def set_twin_message_reactions(
    session: AsyncSession, session_id: str, seq: int, reactions: list[str]
) -> bool:
    """Persist the user's emoji reactions on one persona message. False if unknown."""
    result = await session.execute(
        select(TwinMessage).where(
            TwinMessage.session_id == session_id, TwinMessage.seq == seq
        )
    )
    row = result.scalar_one_or_none()
    if row is None:
        return False
    # Reassign so SQLAlchemy tracks the JSON change.
    row.content = {**(row.content or {}), "reactions": reactions}
    await session.commit()
    return True


async def suggest_twin_personas(
    session: AsyncSession, case_id: str, scan_limit: int = 5
) -> dict[str, object] | None:
    """Suggest a persona line-up from the latest council of the most similar case.

    Degrades to ``None`` when case-similarity (``find_similar_cases``) is not
    available in this build — the UI falls back to the normal picker, and the
    ⭐ recommended personas still come from each persona's ``pinned`` flag.
    """
    find_similar = globals().get("find_similar_cases")
    if find_similar is None:
        return None
    for sim_id, title, pct in await find_similar(session, case_id, limit=scan_limit):
        stmt = (
            select(TwinSession)
            .where(TwinSession.case_id == sim_id)
            .order_by(TwinSession.created_at.desc())
            .limit(1)
        )
        twin_session = (await session.execute(stmt)).scalars().first()
        if twin_session and twin_session.persona_ids:
            return {
                "case_id": sim_id,
                "case_title": title,
                "similarity_pct": pct,
                "persona_ids": twin_session.persona_ids,
            }
    return None


