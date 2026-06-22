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
from sqlalchemy import JSON, DateTime, Integer, String, select, text, update as sa_update
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from llm_wiki.utils.ids import new_file_id

logger = structlog.get_logger(__name__)

_DEV_USER_ID = "dev-user"
_DEV_USER_NAME = "Dev User"
_DEV_USER_ROLE = "admin"


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all metadata models."""


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
