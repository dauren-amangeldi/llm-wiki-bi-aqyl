"""Pydantic request/response models for the API layer."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator  # noqa: F401


class FileUploadResponse(BaseModel):
    """Response body for POST /files.

    ``status="queued"``     — file accepted, pipeline started.
    ``status="duplicate"``  — identical content already exists; no new pipeline run.
    When *status* is ``"duplicate"``, *task_id* is ``None`` and *duplicate_of*
    contains the original ``file_id``.
    """

    file_id: str
    task_id: str | None
    status: Literal["queued", "duplicate"]
    duplicate_of: str | None = None


class StateEntry(BaseModel):
    """A single state transition in the processing history."""

    state: str
    at: datetime

    @field_validator("at", mode="before")
    @classmethod
    def parse_at(cls, v: Any) -> datetime:
        """Accept ISO-8601 strings as well as datetime objects.

        The DB stores ``at`` as a string; Pydantic receives it before
        the normal datetime coercion runs.
        """
        if isinstance(v, str):
            return datetime.fromisoformat(v)
        return v  # type: ignore[return-value]


class FileStatusResponse(BaseModel):
    """Response body for GET /files/{file_id}."""

    file_id: str
    original_name: str
    status: str
    state_history: list[StateEntry]
    created_pages: list[str]
    updated_pages: list[str]
    cost_usd: float | None


class WikiPageResponse(BaseModel):
    """Response body for GET /wiki/{slug} (JSON mode)."""

    slug: str
    title: str
    content: str
    backlinks: list[str]
    last_updated: datetime
    source_files: list[str]


class IssueResponse(BaseModel):
    """A single quality issue (serialised for API responses)."""

    kind: str
    section: str
    page_slug: str
    description: str
    related_slugs: list[str]


class LintRunResponse(BaseModel):
    """Response body for POST /api/v1/lint/run."""

    issues_found: int
    by_kind: dict[str, int]
    issues: list[IssueResponse]
    issues_md_updated: bool


class AuditRunRequest(BaseModel):
    """Request body for POST /api/v1/audit/run."""

    mode: Literal["sync", "batch"] = "batch"
    dry_run: bool = False
    sample: int | None = None
    slugs: list[str] | None = None


class AuditRunResponse(BaseModel):
    """Response body for POST /api/v1/audit/run (202 Accepted)."""

    task_id: str
    mode: str
    estimated_cost_usd: float | None = None
    estimated_completion_at: datetime | None = None


class AuditStatusResponse(BaseModel):
    """Response body for GET /api/v1/audit/{task_id}."""

    task_id: str
    status: str
    result: dict[str, Any] | None = None


class LogResponse(BaseModel):
    """Response body for GET /log."""

    page: int
    per_page: int
    total: int
    entries: list[str]


class StatsResponse(BaseModel):
    """Response body for GET /stats."""

    total_files: int
    total_wiki_pages: int
    cost_today_usd: float
    cost_this_month_usd: float
    avg_cost_per_ingestion_usd: float
    last_lint_run: datetime | None
    # Budget fields (LW-19) — None when no daily limit is configured
    budget_cost_limit_usd: float | None = None
    budget_cost_used_pct: float | None = None


class AskRequest(BaseModel):
    """Request body for POST /api/v1/ask."""

    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=10)


class AskSource(BaseModel):
    """A single source page cited by the AnswerAgent."""

    slug: str
    title: str
    similarity: float


class AskResponse(BaseModel):
    """Response body for POST /api/v1/ask."""

    question: str
    answer: str
    confidence: Literal["high", "medium", "low"]
    sources: list[AskSource]
    cost_usd: float
