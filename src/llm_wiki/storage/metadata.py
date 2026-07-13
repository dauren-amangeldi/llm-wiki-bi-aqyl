"""PostgreSQL metadata store — files, cases, chat history, users, skills.

Uses SQLAlchemy async ORM (psycopg driver). Tables are created on startup via
``Base.metadata.create_all``; schema changes should move to Alembic when needed.
"""

from dataclasses import dataclass
from datetime import UTC, datetime

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
    create_engine,
    select,
)
from sqlalchemy import (
    update as sa_update,
)
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession
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
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
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
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class IssuesReport(Base):
    """Rendered quality-issues sections (replaces issues.md). One row per section."""

    __tablename__ = "issues_report"

    section: Mapped[str] = mapped_column(String, primary_key=True)  # auto-detected | llm-flagged
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class Feedback(Base):
    """A 👍/👎 vote on any AI-produced artefact (similar-case chip, verdict…).

    entity_type + entity_id identify the thing being rated without FKs, so new
    surfaces can start collecting feedback without schema changes. This is the
    cheap signal source the Twins outcome journal builds on.
    """

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    entity_type: Mapped[str] = mapped_column(String, nullable=False, index=True)
    entity_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    vote: Mapped[int] = mapped_column(Integer, nullable=False)  # +1 | -1
    comment: Mapped[str] = mapped_column(String(1000), nullable=False, default="")
    created_by: Mapped[str] = mapped_column(String, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class LLMCallLog(Base):
    """Outcome of every LLM call — success, final failure, or budget block.

    Complements usage.log (successes only): this is where error rate and
    latency percentiles come from. Written synchronously from LLMClient via
    get_sync_engine(); a failed write must never break the LLM call itself.
    """

    __tablename__ = "llm_call_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True
    )
    file_id: Mapped[str] = mapped_column(String, nullable=False, default="")
    agent_type: Mapped[str] = mapped_column(String, nullable=False)
    model: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, index=True)  # ok | error | blocked
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cost_usd: Mapped[float] = mapped_column(nullable=False, default=0.0)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    error_type: Mapped[str] = mapped_column(String, nullable=False, default="")
    error_message: Mapped[str] = mapped_column(String(500), nullable=False, default="")


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
        default=lambda: datetime.now(UTC),
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
        default=lambda: datetime.now(UTC),
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
    state_history: Mapped[list[dict[str, str]]] = mapped_column(JSON, default=list)
    created_pages: Mapped[list[str]] = mapped_column(JSON, default=list)
    updated_pages: Mapped[list[str]] = mapped_column(JSON, default=list)
    cost_usd: Mapped[float | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class CaseRecord(Base):
    """A user-created case (topic container) grouping related documents."""

    __tablename__ = "cases"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    doc_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    # Case-level privacy: new cases start private (sensitive=true) — an
    # employee builds up their own case before explicitly making it public.
    # Files uploaded to a private case are sent to /uploads with
    # sensitive=true so the ingestion pipeline skips inference/indexing into
    # the shared knowledge base (frontend: stores/cases.ts).
    sensitive: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    # Content source, independent of privacy: "internal" (BI Group data) or
    # "external" (world/university/books/public sources). Chosen once by the
    # user when the case is created — not auto-detected.
    source: Mapped[str] = mapped_column(String, nullable=False, default="internal")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


async def ensure_case_columns(conn: AsyncConnection) -> None:
    """Backfill columns added to CaseRecord after the table already existed.

    create_all() only creates missing tables, not missing columns on
    existing ones — needed once for any dev/prod DB created before the
    `sensitive`/`source` columns existed.
    """
    from sqlalchemy import text

    await conn.execute(
        text("ALTER TABLE cases ADD COLUMN IF NOT EXISTS sensitive boolean NOT NULL DEFAULT true")
    )
    await conn.execute(
        text("ALTER TABLE cases ADD COLUMN IF NOT EXISTS source varchar NOT NULL DEFAULT 'internal'")
    )


class CaseEmbedding(Base):
    """Mean-pooled embedding over a case's document chunks.

    Recomputed whenever a case's doc_ids change (see refresh_case_embedding).
    Powers "cases similar to this one" search the same way heading/chunk
    embeddings power wiki search — cosine distance over an HNSW index.
    """

    __tablename__ = "case_embeddings"

    case_id: Mapped[str] = mapped_column(String, primary_key=True)
    embedding: Mapped[list[float]] = mapped_column(Vector(_EMBED_DIM), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    __table_args__ = (
        Index(
            "ix_case_embeddings_vec",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
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
        default=lambda: datetime.now(UTC),
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
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class TwinPersona(Base):
    """A Twins council persona — tech-visionary or development track (BI-AQYL-TWINS).

    Content (name/lens/system_prompt/domain_weights) is sourced from editable
    .md files in ``src/llm_wiki/personas/twins/`` — see ``seed_twin_personas``.
    """

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
    active: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class TwinPreset(Base):
    """A ready-made triad preset for the Twins council (BI-AQYL-TWINS)."""

    __tablename__ = "twin_presets"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    persona_ids: Mapped[list[str]] = mapped_column(JSON, default=list)

class TwinSession(Base):
    """A single Twins council run — which case, which personas (BI-AQYL-TWINS)."""

    __tablename__ = "twin_sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    case_id: Mapped[str] = mapped_column(String, nullable=False)
    persona_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    created_by: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
    )
    # Outcome journal (R2-5): did the council's verdict hold up in reality?
    # "" = not reviewed yet; the calibration data no competitor can copy.
    outcome: Mapped[str] = mapped_column(String, nullable=False, default="")  # "" | confirmed | refuted
    outcome_note: Mapped[str] = mapped_column(String(1000), nullable=False, default="")
    outcome_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class TwinMessage(Base):
    """A single message in a Twins chat transcript (BI-AQYL-TWINS)."""

    __tablename__ = "twin_messages"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    role: Mapped[str] = mapped_column(String, nullable=False)  # user | persona | verdict
    persona_id: Mapped[str | None] = mapped_column(String, nullable=True)
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
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



_DEFAULT_TWIN_PRESET_SEEDS: list[dict[str, object]] = [
    {"id": "preset-tech-transform", "name": "Технотрансформация", "persona_ids": ["musk", "huang", "nadella"]},
    {"id": "preset-should-build", "name": "Стоит ли строить", "persona_ids": ["zell", "bren", "musk"]},
    {"id": "preset-how-to-build", "name": "Каким строить", "persona_ids": ["hines", "alabbar", "huang"]},
    {"id": "preset-sell-adopt", "name": "Как продать / внедрить", "persona_ids": ["corcoran", "miller", "nadella"]},
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
        .values(status=new_status, updated_at=datetime.now(UTC))
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
    history.append({"state": state, "at": datetime.now(UTC).isoformat()})
    await session.execute(
        sa_update(FileRecord)
        .where(FileRecord.file_id == file_id)
        .values(state_history=history, updated_at=datetime.now(UTC))
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

    Strict whitelist: unknown email → denied; ``blocked`` row → denied;
    otherwise allowed with ``is_admin`` taken from the row.
    """
    row = await get_allowed_user(session, email)
    if row is None:
        return AccessDecision(False, False, "not_whitelisted")
    if row.blocked:
        return AccessDecision(False, False, "blocked")
    return AccessDecision(True, bool(row.is_admin), "ok")


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
    values: dict[str, object] = {"updated_at": datetime.now(UTC)}
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



# Twin personas CRUD

async def twin_personas_count(session: AsyncSession) -> int:
    """Return the number of rows in the twin_personas table."""
    result = await session.execute(select(TwinPersona))
    return len(result.scalars().all())


async def get_twin_persona(session: AsyncSession, persona_id: str) -> TwinPersona | None:
    """Return a single persona by id, or None if missing."""
    return await session.get(TwinPersona, persona_id)


async def list_twin_personas(session: AsyncSession) -> list[TwinPersona]:
    """Return all active personas."""
    result = await session.execute(select(TwinPersona).where(TwinPersona.active == 1))
    return list(result.scalars().all())


async def list_twin_presets(session: AsyncSession) -> list[TwinPreset]:
    """Return all preset triads."""
    result = await session.execute(select(TwinPreset))
    return list(result.scalars().all())


async def seed_twin_personas(session: AsyncSession) -> int:
    """Load personas from `.md` files and upsert them into `twin_personas`.

    Upsert (not insert-only) so editing a persona's file and restarting the
    service applies the change without a manual migration. Presets are
    seeded alongside, insert-if-absent (they're structural id-lists, not
    editorial content, so they don't need the upsert treatment).

    Returns the number of persona rows newly inserted (updates aren't counted).
    """
    from llm_wiki.storage.persona_files import load_persona_files

    inserted = 0
    for row in load_persona_files():
        persona_id = str(row["id"])
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

    for preset_row in _DEFAULT_TWIN_PRESET_SEEDS:
        preset_id = str(preset_row["id"])
        if await session.get(TwinPreset, preset_id) is not None:
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

# Twin sessions/messages CRUD

async def create_twin_session(
    session: AsyncSession, *, case_id: str, persona_ids: list[str], created_by: str
) -> TwinSession:
    """Create and persist a new Twins council session."""
    now = datetime.now(UTC)
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
    now = datetime.now(UTC)
    row = TwinMessage(
        id=f"twin-msg-{session_id}-{seq}",
        session_id=session_id,
        role=role,
        persona_id=persona_id,
        seq=seq,
        content=content,
        created_at=now,
    )
    session.add(row)
    await session.commit()
    return row


async def get_twin_session_messages(session: AsyncSession, session_id: str) -> list[TwinMessage]:
    """Return all messages for a session, ordered by seq."""
    result = await session.execute(
        select(TwinMessage).where(TwinMessage.session_id == session_id).order_by(TwinMessage.seq)
    )
    return list(result.scalars().all())




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


async def refresh_case_embedding(session: AsyncSession, case_id: str) -> bool:
    """Recompute and upsert a case's mean-pooled embedding from its documents' chunks.

    Call this whenever a case's doc_ids change. A case created with documents
    that are still mid-ingestion (no chunk embeddings yet) simply gets no
    embedding until the next refresh — known limitation, not handled by a
    pipeline hook yet.

    Args:
        session: Active async SQLAlchemy session.
        case_id: The case to refresh.

    Returns:
        True if an embedding was computed and stored, False if the case has
        no documents yet, or none of them have chunk embeddings yet.
    """
    case = await session.get(CaseRecord, case_id)
    if not case or not case.doc_ids:
        return False

    rows = (
        await session.execute(
            select(ChunkEmbedding.embedding).where(ChunkEmbedding.file_id.in_(case.doc_ids))
        )
    ).scalars().all()
    if not rows:
        return False

    dim = len(rows[0])
    mean = [sum(v[i] for v in rows) / len(rows) for i in range(dim)]

    existing = await session.get(CaseEmbedding, case_id)
    if existing:
        existing.embedding = mean
    else:
        session.add(CaseEmbedding(case_id=case_id, embedding=mean))
    await session.commit()
    return True


async def find_similar_cases(
    session: AsyncSession, case_id: str, limit: int = 5
) -> list[tuple[str, str, float]]:
    """Find cases most similar to *case_id* by cosine distance over case embeddings.

    Args:
        session: Active async SQLAlchemy session.
        case_id: The case to compare against.
        limit: Max number of matches to return.

    Returns:
        List of (case_id, title, similarity_pct) tuples, most similar first.
        Empty if *case_id* has no embedding yet (no docs, or docs still
        processing).
    """
    target = await session.get(CaseEmbedding, case_id)
    if not target:
        return []

    distance = CaseEmbedding.embedding.cosine_distance(target.embedding)
    stmt = (
        select(CaseRecord.id, CaseRecord.title, distance.label("distance"))
        .join(CaseEmbedding, CaseEmbedding.case_id == CaseRecord.id)
        .where(CaseEmbedding.case_id != case_id)
        .order_by(distance)
        .limit(limit)
    )
    rows = (await session.execute(stmt)).all()
    return [(r.id, r.title, round((1 - r.distance) * 100, 1)) for r in rows]


async def list_twin_sessions(session: AsyncSession, case_id: str) -> list[TwinSession]:
    """Past councils for a case, newest first (outcome journal view)."""
    stmt = (
        select(TwinSession)
        .where(TwinSession.case_id == case_id)
        .order_by(TwinSession.created_at.desc())
    )
    return list((await session.execute(stmt)).scalars().all())


async def set_twin_session_outcome(
    session: AsyncSession, session_id: str, outcome: str, note: str = ""
) -> bool:
    """Record whether a council's verdict held up. False if the session is unknown."""
    row = await session.get(TwinSession, session_id)
    if not row:
        return False
    row.outcome = outcome
    row.outcome_note = note
    row.outcome_at = datetime.now(UTC)
    await session.commit()
    return True


async def suggest_twin_personas(
    session: AsyncSession, case_id: str, scan_limit: int = 5
) -> dict[str, object] | None:
    """Suggest a Twins persona line-up based on the most similar past case.

    Walks similar cases (most similar first) and returns the persona set of
    the latest council held on the first one that has any. None when nothing
    similar has ever been deliberated — the UI just shows the normal picker.
    """
    for sim_id, title, pct in await find_similar_cases(session, case_id, limit=scan_limit):
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


def log_llm_call(
    *,
    file_id: str,
    agent_type: str,
    model: str,
    status: str,
    duration_ms: int,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_usd: float = 0.0,
    attempts: int = 1,
    error: Exception | None = None,
) -> None:
    """Record an LLM call outcome in llm_call_log. Never raises.

    Sync on purpose: called from both async complete() and sync embed() in
    LLMClient without event-loop acrobatics. The insert takes ~1 ms against
    a local Postgres — negligible next to a multi-second LLM call.
    """
    from sqlalchemy.orm import Session as SyncSession

    try:
        with SyncSession(get_sync_engine()) as s:
            s.add(
                LLMCallLog(
                    file_id=file_id,
                    agent_type=agent_type,
                    model=model,
                    status=status,
                    duration_ms=duration_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost_usd,
                    attempts=attempts,
                    error_type=type(error).__name__ if error else "",
                    error_message=str(error)[:500] if error else "",
                )
            )
            s.commit()
    except Exception as exc:  # noqa: BLE001 — telemetry must not break the call
        logger.warning("llm_call_log_write_failed", error=str(exc))


