"""Studio artifact generation — test / report / presentation / card / infographic
generated from a case or document's wiki content via the LLM.

Grounding mirrors ``AnswerAgent``: load the wiki pages the source produced and
feed them as context. Each text artifact is one JSON LLM call whose shape matches
the frontend renderer; the infographic is a template SVG built from a compact
LLM summary (never "let the LLM draw SVG").
"""

from __future__ import annotations

import html
import json
from typing import Any

import structlog

from llm_wiki.llm.client import LLMClient
from llm_wiki.storage import wiki_store
from llm_wiki.storage.metadata import CaseRecord, FileRecord, get_file_record

logger = structlog.get_logger(__name__)

_MAX_PAGE_CHARS = 4000
_MAX_TOTAL_CHARS = 20_000

# Frontend keys → backend kinds (the renderers expect these exact kinds).
GENERATED_KINDS = frozenset({"report", "card", "test", "presentation", "infographic"})


class ArtifactError(Exception):
    """Raised when an artifact can't be generated (no content, bad LLM output)."""


# --- JSON schemas (match the renderers; strict-friendly) ---------------------

def _obj(props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }


_REPORT_SCHEMA = _obj(
    {
        "executive_summary": {"type": "string"},
        "metrics": {
            "type": "array",
            "items": _obj({"label": {"type": "string"}, "value": {"type": "string"}}, ["label", "value"]),
        },
        "sections": {
            "type": "array",
            "items": _obj({"heading": {"type": "string"}, "body": {"type": "string"}}, ["heading", "body"]),
        },
    },
    ["executive_summary", "metrics", "sections"],
)

_CARDS_SCHEMA = _obj(
    {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "key_points": {
            "type": "array",
            "items": _obj({"label": {"type": "string"}, "text": {"type": "string"}}, ["label", "text"]),
        },
        "recommendations": {"type": "array", "items": {"type": "string"}},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    ["title", "summary", "key_points", "recommendations", "tags"],
)

_TEST_SCHEMA = _obj(
    {
        "questions": {
            "type": "array",
            "items": _obj(
                {
                    "prompt": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"}},
                    "correct": {"type": "integer"},
                    "explanation": {"type": "string"},
                },
                ["prompt", "options", "correct", "explanation"],
            ),
        }
    },
    ["questions"],
)

_PRESENTATION_SCHEMA = _obj(
    {
        "title": {"type": "string"},
        "slides": {
            "type": "array",
            "items": _obj(
                {
                    "heading": {"type": "string"},
                    "bullets": {"type": "array", "items": {"type": "string"}},
                    "notes": {"type": "string"},
                },
                ["heading", "bullets", "notes"],
            ),
        },
    },
    ["title", "slides"],
)

_INFOGRAPHIC_SCHEMA = _obj(
    {
        "title": {"type": "string"},
        "stats": {
            "type": "array",
            "items": _obj({"label": {"type": "string"}, "value": {"type": "string"}}, ["label", "value"]),
        },
        "points": {"type": "array", "items": {"type": "string"}},
    },
    ["title", "stats", "points"],
)

_PROMPT_BY_KIND = {
    "report": ("artifact_report", _REPORT_SCHEMA, "report"),
    "card": ("artifact_cards", _CARDS_SCHEMA, "cards"),
    "test": ("artifact_test", _TEST_SCHEMA, "test"),
    "presentation": ("artifact_presentation", _PRESENTATION_SCHEMA, "presentation"),
    "infographic": ("artifact_infographic", _INFOGRAPHIC_SCHEMA, "infographic"),
}


# --- Source gathering --------------------------------------------------------

async def _title_and_slugs(session: Any, document_id: str) -> tuple[str, list[str]]:
    """Return (title, wiki slugs) for a case id or a single document id."""
    case = await session.get(CaseRecord, document_id)
    if case is not None:
        slugs: list[str] = []
        for did in case.doc_ids or []:
            fr = await session.get(FileRecord, did)
            if fr is not None:
                slugs.extend(list(fr.created_pages or []))
                slugs.extend(list(fr.updated_pages or []))
        return case.title, list(dict.fromkeys(slugs))

    fr = await get_file_record(session, document_id)
    if fr is not None:
        slugs = list(dict.fromkeys(list(fr.created_pages or []) + list(fr.updated_pages or [])))
        return fr.original_name, slugs
    return document_id, []


def _load_bodies(slugs: list[str]) -> str:
    parts: list[str] = []
    total = 0
    for slug in slugs:
        try:
            body = wiki_store.get_page(slug) or ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("artifact_load_page_failed", slug=slug, error=str(exc))
            body = ""
        if not body:
            continue
        chunk = body[:_MAX_PAGE_CHARS]
        if total + len(chunk) > _MAX_TOTAL_CHARS:
            break
        parts.append(chunk)
        total += len(chunk)
    return "\n\n---\n\n".join(parts)


# --- Generation --------------------------------------------------------------

async def generate_content(
    session: Any,
    llm: LLMClient,
    *,
    kind: str,
    document_id: str,
    language: str,
) -> dict[str, Any]:
    """Generate an artifact's content dict for the given kind (grounded in sources)."""
    if kind not in _PROMPT_BY_KIND:
        raise ArtifactError(f"Unsupported artifact kind: {kind!r}")

    title, slugs = await _title_and_slugs(session, document_id)
    content_text = _load_bodies(slugs)
    if not content_text.strip():
        raise ArtifactError("No source content available to generate from.")

    prompt_name, schema, schema_name = _PROMPT_BY_KIND[kind]
    prompt = llm.load_prompt(
        prompt_name, language=language, title=title, content=content_text
    )
    text, _usage = await llm.complete(
        prompt=prompt,
        system="You are a precise studio-artifact generator. Return only valid JSON.",
        file_id=f"artifact-{kind}-{document_id}",
        agent_type="artifact",
        response_format="json",
        json_schema=schema,
        schema_name=schema_name,
    )
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ArtifactError(f"Artifact LLM returned invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ArtifactError("Artifact LLM output was not a JSON object.")

    if kind == "infographic":
        return {"svg": _render_infographic_svg(data)}
    return data


# --- Infographic SVG template (built from LLM summary, not LLM-drawn) ---------

def _render_infographic_svg(data: dict[str, Any]) -> str:
    """Render a clean, self-contained infographic SVG from {title, stats, points}."""
    title = html.escape(str(data.get("title") or "")[:120])
    stats = [s for s in (data.get("stats") or []) if isinstance(s, dict)][:3]
    points = [str(p) for p in (data.get("points") or []) if str(p).strip()][:6]

    w, h = 900, 560
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'width="100%" role="img" aria-label="{title}" font-family="Inter, system-ui, sans-serif">',
        f'<rect x="0" y="0" width="{w}" height="{h}" rx="20" fill="#f6f8fb"/>',
        f'<rect x="0" y="0" width="{w}" height="88" rx="20" fill="#0058ff"/>',
        f'<rect x="0" y="60" width="{w}" height="28" fill="#0058ff"/>',
        f'<text x="40" y="56" fill="#ffffff" font-size="30" font-weight="700">{title}</text>',
    ]

    # KPI cards
    if stats:
        n = len(stats)
        gap = 24
        card_w = (w - 80 - gap * (n - 1)) / n
        for i, s in enumerate(stats):
            x = 40 + i * (card_w + gap)
            value = html.escape(str(s.get("value") or "")[:16])
            label = html.escape(str(s.get("label") or "")[:40])
            parts.append(f'<rect x="{x:.0f}" y="120" width="{card_w:.0f}" height="120" rx="14" fill="#ffffff" stroke="#e6ecf5"/>')
            parts.append(f'<text x="{x + card_w / 2:.0f}" y="180" fill="#0058ff" font-size="34" font-weight="800" text-anchor="middle">{value}</text>')
            parts.append(f'<text x="{x + card_w / 2:.0f}" y="210" fill="#5b6b82" font-size="15" text-anchor="middle">{label}</text>')

    # Bullet points
    y = 290 if stats else 140
    for p in points:
        text = html.escape(p[:110])
        parts.append(f'<circle cx="52" cy="{y - 5:.0f}" r="5" fill="#0058ff"/>')
        parts.append(f'<text x="72" y="{y:.0f}" fill="#1a2740" font-size="18">{text}</text>')
        y += 40

    parts.append("</svg>")
    return "".join(parts)
