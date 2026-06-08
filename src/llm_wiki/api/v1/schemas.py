"""Pydantic schemas for the v1 API adapter.

Match 1:1 with TypeScript interfaces in llm-wiki-frontend/src/stores/materials.ts
and llm-wiki-frontend/src/stores/modal.ts.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class Tag(BaseModel):
    id: str
    name: str


class Material(BaseModel):
    document_id: str
    title: str
    content_type: Literal["pdf", "markdown", "video", "audio"]
    scope: Literal["internal", "external"]
    business_unit: str
    status: str
    created_at: str
    updated_at: str | None = None
    source_language: str | None = None
    tags: list[Tag] = []
    topic_ids: list[str] = []
    title_i18n: dict[str, str] = {}
    snippet: str | None = None
    author: str | None = None
    language: str | None = None
    classification: str | None = None


class MaterialSource(BaseModel):
    title: str
    content_type: str
    path: str | None = None
    document_id: str | None = None
    status: str | None = None


class SourcesResponse(BaseModel):
    items: list[MaterialSource]


class Dossier(BaseModel):
    summary: str | None = None
    page_count: int | None = None
    language: str | None = None
    status: str | None = None
