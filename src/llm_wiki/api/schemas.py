"""Pydantic request/response models for the API layer."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class FileUploadResponse(BaseModel):
    """Response body for POST /files."""

    file_id: str
    task_id: str
    status: Literal["queued"]


class StateEntry(BaseModel):
    """A single state transition in the processing history."""

    state: str
    at: datetime


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


class LintRunResponse(BaseModel):
    """Response body for POST /lint/run."""

    task_id: str
    status: Literal["queued"]


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
