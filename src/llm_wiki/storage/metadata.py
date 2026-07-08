"""PostgreSQL metadata store — files, cases, chat history, users, skills.

Uses SQLAlchemy async ORM (psycopg driver). Tables are created on startup via
``Base.metadata.create_all``; schema changes should move to Alembic when needed.
"""

from datetime import datetime, timezone

import structlog
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    DateTime,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    select,
    update as sa_update,
)
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from llm_wiki.config import settings


logger = structlog.get_logger(__name__)

_DEV_USER_ID = "dev-user"
_DEV_USER_NAME = "Dev User"
_DEV_USER_ROLE = "admin"


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
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


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
        model_name=model_name,
    )
    session.add(record)
    await session.commit()
    return record


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


