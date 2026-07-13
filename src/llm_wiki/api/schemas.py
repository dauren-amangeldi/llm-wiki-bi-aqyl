"""Pydantic request/response models for the API layer."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator  # noqa: F401


class CurrentUser(BaseModel):
    """Authenticated (or dev-stub) user resolved by ``get_current_user`` (LW-N1)."""

    id: str
    name: str
    role: str


class DocumentSearchResult(BaseModel):
    """Document hit returned by GET /search (MVP mock contract)."""

    document_id: str
    document_title: str
    snippet: str = ""
    scope: str = "internal"
    classification: str = ""
    score: float = 1.0
    content_type: str = "markdown"


class WikiSearchResult(BaseModel):
    """Lexical FTS hit returned by GET /search (LW-N5)."""

    slug: str
    title: str
    snippet: str
    scope: str = "wiki"


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


class AdvisorHistoryTurn(BaseModel):
    """One turn in an advisor follow-up conversation."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class AdvisorRequest(BaseModel):
    """Request body for POST /api/v1/advisor."""

    query: str = Field(min_length=3, max_length=1000)
    role: str = Field(default="employee", max_length=64)
    language: str = Field(default="ru", pattern="^(ru|en|kk)$")
    scope: str = Field(default="all", max_length=32)
    history: list[AdvisorHistoryTurn] = Field(default_factory=list)
    session_id: str | None = None


class AdvisorPointResponse(BaseModel):
    """A single insight point in the advisor SSE final event."""

    heading: str
    body: str
    metric: str = ""
    tag: str = ""
    case_id: str


class AdvisorResponseBody(BaseModel):
    """Structured advisor payload (non-refusal)."""

    title: str
    summary: str
    points: list[AdvisorPointResponse]
    source: str
    caseCount: int


class SkillResponse(BaseModel):
    """Skill row exposed to the frontend skills panel (LW-N12)."""

    slug: str
    name: str
    content: str
    role: str
    active: bool
    description: str = ""


class SkillUpdateRequest(BaseModel):
    """Request body for PUT /api/v1/skills/{role}."""

    content: str | None = Field(default=None, max_length=8000)
    system_prompt: str | None = Field(default=None, max_length=8000)
    active: int | None = Field(default=None, ge=0, le=1)

    def resolved_system_prompt(self) -> str | None:
        """Return the prompt field supplied by the client."""
        if self.system_prompt is not None:
            return self.system_prompt
        return self.content


# ── AI-советник: стратегическая консультация ──────────────────────────

class ConsultationStartRequest(BaseModel):
    """Тело POST /api/v1/advisor/consultations."""

    query: str = Field(min_length=3, max_length=2000)
    role: str = Field(default="employee", max_length=64)
    language: str = Field(default="ru", pattern="^(ru|en|kk)$")


class ClarificationQuestion(BaseModel):
    id: str
    text: str
    why_it_matters: str = ""
    options: list[str] = Field(default_factory=list)
    allow_custom: bool = True
    required: bool = False


class ClarificationRequiredResponse(BaseModel):
    mode: Literal["clarification_required"] = "clarification_required"
    session_id: str
    decision_type: str
    questions: list[ClarificationQuestion]
    question_limit: int = 5


class QuestionAnswer(BaseModel):
    question_id: str
    answer: str = ""
    skipped: bool = False


class ConsultationRespondRequest(BaseModel):
    """Тело POST /api/v1/advisor/consultations/{id}/respond."""

    answers: list[QuestionAnswer] = Field(default_factory=list)
    give_advice_now: bool = False


class UnderstandingSnapshot(BaseModel):
    decision: str
    desired_outcome: str
    horizon: str
    constraints: list[str] = Field(default_factory=list)
    stakeholders: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class UnderstandingSnapshotResponse(BaseModel):
    mode: Literal["understanding_snapshot"] = "understanding_snapshot"
    session_id: str
    snapshot: UnderstandingSnapshot


class ConsultationSnapshotUpdate(BaseModel):
    """Тело PUT /api/v1/advisor/consultations/{id}/snapshot — частичное обновление."""

    decision: str | None = None
    desired_outcome: str | None = None
    horizon: str | None = None
    constraints: list[str] | None = None
    stakeholders: list[str] | None = None
    success_criteria: list[str] | None = None
    assumptions: list[str] | None = None


class DecisionBrief(BaseModel):
    recommendation: str
    why_now: str
    problem_frame: str
    key_assumption: str
    rationale: str
    alternatives: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    first_step: str
    reconsider_if: list[str] = Field(default_factory=list)
    evidence_strength: Literal["high", "medium", "low"]
    assumptions: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


class DecisionBriefResponse(BaseModel):
    mode: Literal["decision_brief"] = "decision_brief"
    session_id: str
    brief: DecisionBrief


class ConsultationOutcomeRequest(BaseModel):
    """Тело POST /api/v1/advisor/consultations/{id}/outcome — лёгкая фиксация результата."""

    outcome: Literal["decided", "need_info", "postponed", "rejected"]
