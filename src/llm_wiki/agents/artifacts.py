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


# Reference structure: Резюме / Ключевой вывод / Риски / Рекомендации + chips
# (релевантность %, покрытие цитатами %, горизонт эффекта) + язык оригинала.
# «Источники» и «~мин чтения» заполняются программно (не доверяем их LLM).
_REPORT_SCHEMA = _obj(
    {
        "summary": {"type": "string"},
        "key_insight": {"type": "string"},
        "risks": {"type": "array", "items": {"type": "string"}},
        "recommendations": {"type": "array", "items": {"type": "string"}},
        "relevance_pct": {"type": "integer"},
        "citation_coverage_pct": {"type": "integer"},
        "effect_horizon": {"type": "string"},
        "source_language": {"type": "string"},
    },
    [
        "summary",
        "key_insight",
        "risks",
        "recommendations",
        "relevance_pct",
        "citation_coverage_pct",
        "effect_horizon",
        "source_language",
    ],
)

# Reference structure — a horizontal deck of typed cards:
# ИНСАЙТ / КОНТЕКСТ / ШАГ 1..N / РИСК / ДЕЙСТВИЕ (badges: %, язык, 01.., ⚠, N′).
_CARDS_SCHEMA = _obj(
    {
        "insight": {"type": "string"},
        "context": {"type": "string"},
        "steps": {
            "type": "array",
            "items": _obj({"title": {"type": "string"}, "text": {"type": "string"}}, ["title", "text"]),
        },
        "risk": {"type": "string"},
        "action": {"type": "string"},
        "action_minutes": {"type": "integer"},
        "relevance_pct": {"type": "integer"},
        "source_language": {"type": "string"},
    },
    [
        "insight",
        "context",
        "steps",
        "risk",
        "action",
        "action_minutes",
        "relevance_pct",
        "source_language",
    ],
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

# Reference: a generated infographic PICTURE (OpenAI Images) on top + info cards
# below — eyebrow (direction), key insight (ЦИФРА-ГЕРОЙ), implementation path,
# and stat cards (релевантность / источников / шагов / язык). `image_prompt`
# drives the picture; the structured fields drive the cards.
_INFOGRAPHIC_SCHEMA = _obj(
    {
        "eyebrow": {"type": "string"},
        "headline": {"type": "string"},
        "key_insight": {"type": "string"},
        "stats": {
            "type": "array",
            "items": _obj(
                {"label": {"type": "string"}, "value": {"type": "string"}},
                ["label", "value"],
            ),
        },
        "implementation_path": {"type": "array", "items": {"type": "string"}},
        "relevance_pct": {"type": "integer"},
        "source_language": {"type": "string"},
    },
    [
        "eyebrow", "headline", "key_insight", "stats",
        "implementation_path", "relevance_pct", "source_language",
    ],
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


def _page_title(slug: str, body: str) -> str:
    """First markdown H1 of the page, else the slug prettified."""
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
        if stripped:
            break
    return slug.replace("-", " ").replace("_", " ").strip() or slug


def _load_bodies(slugs: list[str]) -> tuple[str, list[str]]:
    """Concatenated (truncated) page bodies + titles of the pages actually used."""
    parts: list[str] = []
    titles: list[str] = []
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
        titles.append(_page_title(slug, body))
        total += len(chunk)
    return "\n\n---\n\n".join(parts), titles


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
    content_text, source_titles = _load_bodies(slugs)
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
        # Structured fields drive the HTML info cards the frontend renders under
        # the picture (relevance / sources / steps + ЦИФРА-ГЕРОЙ + source line).
        fields: dict[str, Any] = {
            "title": title,
            "eyebrow": str(data.get("eyebrow") or ""),
            "headline": str(data.get("headline") or ""),
            "key_insight": str(data.get("key_insight") or ""),
            "stats": [
                {"label": str(s.get("label") or ""), "value": str(s.get("value") or "")}
                for s in (data.get("stats") or [])
                if isinstance(s, dict)
            ][:3],
            "implementation_path": [str(s) for s in (data.get("implementation_path") or [])],
            "relevance_pct": _clamp_pct(data.get("relevance_pct")),
            "source_language": str(data.get("source_language") or ""),
            "sources_count": max(1, len(source_titles)),
            "source_line": " · ".join(source_titles[:3]) or title,
        }
        # Primary: a generated picture (OpenAI Images, text-free prompt). Fall
        # back to the self-contained SVG when image generation is unavailable
        # (no model access, API error) so the artifact always renders.
        try:
            image_url = await llm.generate_image(_infographic_image_prompt(data, title))
            return {"image_url": image_url, **fields}
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "infographic_image_failed", error=str(exc), document_id=document_id
            )
            svg = _render_infographic_svg(data, title=title, source_titles=source_titles)
            return {"svg": svg, **fields}
    if kind == "report":
        return _finalize_report(data, source_titles)
    if kind == "card":
        return _finalize_cards(data)
    return data


def _clamp_pct(value: Any) -> int:
    try:
        return max(0, min(100, int(value)))
    except (TypeError, ValueError):
        return 0


def _finalize_cards(data: dict[str, Any]) -> dict[str, Any]:
    """Sanity clamps for the card deck (badges must stay renderable)."""
    data["relevance_pct"] = _clamp_pct(data.get("relevance_pct"))
    try:
        minutes = int(data.get("action_minutes"))
    except (TypeError, ValueError):
        minutes = 30
    data["action_minutes"] = max(5, min(480, minutes))
    steps = [s for s in data.get("steps") or [] if isinstance(s, dict)]
    data["steps"] = steps[:6]
    return data


def _finalize_report(data: dict[str, Any], source_titles: list[str]) -> dict[str, Any]:
    """Programmatic report fields: real sources, clamped %, reading time."""
    data["sources"] = source_titles
    data["relevance_pct"] = _clamp_pct(data.get("relevance_pct"))
    data["citation_coverage_pct"] = _clamp_pct(data.get("citation_coverage_pct"))
    text_parts = [
        str(data.get("summary") or ""),
        str(data.get("key_insight") or ""),
        *[str(r) for r in data.get("risks") or []],
        *[str(r) for r in data.get("recommendations") or []],
    ]
    words = len(" ".join(text_parts).split())
    data["reading_minutes"] = max(1, round(words / 170))
    return data


# --- Infographic SVG template (built from LLM summary, not LLM-drawn) ---------

# Palette (light, "email/dashboard" graphic — intentionally not theme-aware).
_IG_GOLD = "#B7791F"
_IG_INK = "#1A2233"
_IG_MUTED = "#64748B"
_IG_CARD = "#F3F5FB"
_IG_BORDER = "#E6ECF5"


_IG_LANG_NAME = {"RU": "Russian", "EN": "English", "KK": "Kazakh"}

# The generated-picture palette (tweak here to restyle the infographic image).
_IG_IMAGE_PALETTE = (
    "a bright royal/cobalt blue as the primary colour (vivid and saturated — NOT a "
    "pale sky-blue or cyan) with warm orange accents, on a clean white background"
)


def _infographic_image_prompt(data: dict[str, Any], title: str) -> str:
    """Build a DATA infographic prompt: embed the real figures, steps and title so
    gpt-image-1 draws an informative infographic (numbers, labels, charts) — not a
    decorative illustration. Text is requested in the source language, spelled as
    given; the HTML cards below stay the accurate source of truth."""
    lang = _IG_LANG_NAME.get(str(data.get("source_language") or "").upper(), "Russian")
    headline = str(data.get("headline") or data.get("eyebrow") or title or "").strip()
    tagline = str(data.get("eyebrow") or "").strip()
    stats = [s for s in (data.get("stats") or []) if isinstance(s, dict)][:3]
    steps = [str(s).strip() for s in (data.get("implementation_path") or []) if str(s).strip()][:4]

    stat_lines = "\n".join(
        f'   - big number "{str(s.get("value", "")).strip()}" with the caption '
        f'"{str(s.get("label", "")).strip()}"'
        for s in stats
    ) or "   - three key KPI numbers with captions"
    step_lines = ", ".join(f'"{s}"' for s in steps) or "four short steps"

    return (
        "Design a professional full-page CORPORATE INFOGRAPHIC POSTER, portrait "
        "orientation, modern flat vector style, clean, premium and trustworthy. "
        f"Palette: {_IG_IMAGE_PALETTE}. "
        "Lay it out top-to-bottom like a polished business report:\n"
        f'1) A full-width colored HEADER BAND with the bold title "{headline}"'
        f'{f" and a short subtitle “{tagline}”" if tagline else ""} in white, '
        "plus a small upward-trend chart icon on the right.\n"
        "2) A row of THREE KPI STATS, each a circular flat icon above a large bold "
        f"number and a one-word caption:\n{stat_lines}\n"
        f"3) A process section (write its heading in {lang}, e.g. «{len(steps) or 4} шага»): "
        "four connected chevron/arrow blocks numbered 01–04, each with a simple icon "
        f"and a short label ({step_lines}) and a tiny one-line caption under it.\n"
        "4) Two data charts side by side: a vertical BAR CHART with an axis and value "
        "labels, and a DONUT CHART with a small legend and percentage labels.\n"
        "5) A full-width colored FOOTER BAND with a small globe icon and a short source line.\n"
        f"Write ALL text and numbers in {lang}, spelled EXACTLY as given, crisp and "
        "legible. Balanced grid, clear hierarchy, generous whitespace, high detail — a "
        "real data infographic like a corporate annual-report one-pager."
    )


def _wrap_words(text: str, max_chars: int, max_lines: int) -> list[str]:
    """Greedy word-wrap to ~max_chars per line (SVG <text> doesn't wrap)."""
    lines: list[str] = []
    cur = ""
    for word in text.split():
        if cur and len(cur) + 1 + len(word) > max_chars:
            lines.append(cur)
            cur = word
            if len(lines) == max_lines - 1:
                break
        else:
            cur = f"{cur} {word}".strip()
    rest = text.split()
    if len(lines) == max_lines - 1:  # dump whatever's left onto the final line
        used = sum(len(line_.split()) for line_ in lines)
        cur = " ".join(rest[used:])
    if cur:
        lines.append(cur)
    if lines and len(lines) == max_lines and len(cur) > max_chars:
        lines[-1] = cur[: max_chars - 1].rstrip() + "…"
    return lines[:max_lines]


def _svg_lines(x: int, y: int, lines: list[str], size: int, lh: int, color: str, weight: int) -> str:
    tspans = "".join(
        f'<tspan x="{x}" y="{y + i * lh}">{html.escape(s)}</tspan>' for i, s in enumerate(lines)
    )
    return f'<text fill="{color}" font-size="{size}" font-weight="{weight}">{tspans}</text>'


def _render_infographic_svg(
    data: dict[str, Any], *, title: str, source_titles: list[str]
) -> str:
    """Render the one-screen "визуальная сводка" SVG (reference layout):
    eyebrow + title + key insight + implementation path, and a 2×2 stat grid.
    Self-contained + shareable (email/dashboard)."""
    eyebrow = html.escape(str(data.get("eyebrow") or "").upper()[:40])
    insight = str(data.get("key_insight") or "")
    path_steps = [str(s).strip() for s in (data.get("implementation_path") or []) if str(s).strip()][:6]
    relevance = _clamp_pct(data.get("relevance_pct"))
    lang = html.escape(str(data.get("source_language") or "").upper()[:4] or "—")
    sources_count = max(1, len(source_titles))
    steps_count = len(path_steps)
    source_line = html.escape((" · ".join(source_titles[:3]) or title)[:90])
    path_str = html.escape(" → ".join(path_steps)[:80])

    w, h = 1000, 600
    p: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="100%" '
        f'role="img" aria-label="{html.escape(title[:80])}" '
        f'font-family="Inter, -apple-system, Segoe UI, Roboto, sans-serif">',
        f'<rect x="8" y="8" width="{w - 16}" height="{h - 16}" rx="22" fill="#ffffff" stroke="{_IG_BORDER}"/>',
        f'<line x1="620" y1="56" x2="620" y2="{h - 56}" stroke="{_IG_BORDER}"/>',
    ]

    # ── Left column ──
    p.append(f'<text x="52" y="78" fill="{_IG_GOLD}" font-size="14" font-weight="700" letter-spacing="1.5">{eyebrow}</text>')
    p.append(_svg_lines(52, 118, _wrap_words(title, 26, 2), 32, 42, _IG_INK, 800))

    p.append(f'<text x="52" y="264" fill="{_IG_MUTED}" font-size="13" font-weight="700" letter-spacing="1.2">ГЛАВНЫЙ ИНСАЙТ</text>')
    p.append(_svg_lines(52, 296, _wrap_words(insight, 36, 4), 22, 31, _IG_GOLD, 700))

    p.append(f'<text x="52" y="472" fill="{_IG_MUTED}" font-size="13" font-weight="700" letter-spacing="1.2">ПУТЬ ВНЕДРЕНИЯ</text>')
    p.append(f'<text x="52" y="506" fill="{_IG_INK}" font-size="20" font-weight="700">{path_str}</text>')

    p.append(f'<text x="52" y="{h - 40}" fill="{_IG_MUTED}" font-size="13">{source_line}</text>')

    # ── Right column: 2×2 stat cards ──
    cards = [
        (f"{relevance}%", "Релевантность"),
        (str(sources_count), "Источников"),
        (str(steps_count), "Шагов внедрения"),
        (lang, "Язык оригинала"),
    ]
    cw, ch, gap, x0, y0 = 150, 96, 16, 656, 60
    for i, (value, label) in enumerate(cards):
        cx = x0 + (i % 2) * (cw + gap)
        cy = y0 + (i // 2) * (ch + gap)
        p.append(f'<rect x="{cx}" y="{cy}" width="{cw}" height="{ch}" rx="14" fill="{_IG_CARD}" stroke="{_IG_BORDER}"/>')
        p.append(f'<text x="{cx + 20}" y="{cy + 46}" fill="{_IG_GOLD}" font-size="30" font-weight="800">{html.escape(value)}</text>')
        p.append(f'<text x="{cx + 20}" y="{cy + 74}" fill="{_IG_MUTED}" font-size="14">{label}</text>')

    p.append("</svg>")
    return "".join(p)
