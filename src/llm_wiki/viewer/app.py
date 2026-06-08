"""LLM Wiki Viewer — read-only Streamlit browser.

Reads directly from the filesystem (no API calls).  Mount data/ as read-only
via Docker volume; set DATA_DIR env var if the path differs.

Navigation model: st.session_state["nav"] ∈ {"index","page","changelog","stats"}
                  st.session_state["slug"] holds the current wiki page slug.
"""

import os
import re
from datetime import datetime
from pathlib import Path

import httpx
import streamlit as st

from llm_wiki.ui.markdown_safe import escape_dollars_for_streamlit

# ---------------------------------------------------------------------------
# Paths (configurable via environment for Docker flexibility)
# ---------------------------------------------------------------------------

DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
WIKI_DIR = DATA_DIR / "wiki"
INDEX_PATH = DATA_DIR / "index.md"
LOG_PATH = DATA_DIR / "log.md"
RAW_DIR = DATA_DIR / "raw"

API_BASE = os.getenv("API_BASE_URL", "http://api:8000")

# ---------------------------------------------------------------------------
# i18n for the Ask page (Patch 4 — Russian / Kazakh / English)
# ---------------------------------------------------------------------------

_LANG = os.getenv("WIKI_LANGUAGE", "ru").lower()

_UI: dict[str, str] = {
    "ru": {
        "title": "❓ Спросить вики",
        "caption": "Ответ синтезируется только из проиндексированных страниц.",
        "input_label": "Ваш вопрос",
        "placeholder": "например, Что такое LoRA?",
        "slider": "Сколько страниц-источников учитывать",
        "btn_ask": "Спросить",
        "thinking": "Думаю...",
        "confidence_label": "Уверенность",
        "cost_label": "Стоимость",
        "sources": "Источники",
        "no_sources": "Источники не использованы — скорее всего, тема не покрыта в вики.",
        "btn_sidebar": "❓ Спросить",
    },
    "kk": {
        "title": "❓ Уикиден сұрау",
        "caption": "Жауап тек индекстелген беттер негізінде құрастырылады.",
        "input_label": "Сіздің сұрағыңыз",
        "placeholder": "мысалы, LoRA дегеніміз не?",
        "slider": "Қанша бет-дерек көзін ескеру керек",
        "btn_ask": "Сұрау",
        "thinking": "Ойланудамын...",
        "confidence_label": "Сенімділік",
        "cost_label": "Құны",
        "sources": "Дереккөздер",
        "no_sources": "Дереккөздер пайдаланылмады — уикиде осы тақырып жоқ сияқты.",
        "btn_sidebar": "❓ Сұрау",
    },
    "en": {
        "title": "❓ Ask the wiki",
        "caption": "Answers are synthesised only from indexed wiki pages.",
        "input_label": "Your question",
        "placeholder": "e.g. What is LoRA?",
        "slider": "How many source pages to consider",
        "btn_ask": "Ask",
        "thinking": "Thinking...",
        "confidence_label": "Confidence",
        "cost_label": "Cost",
        "sources": "Sources",
        "no_sources": "No sources were used. The wiki likely does not cover this topic.",
        "btn_sidebar": "❓ Ask",
    },
}.get(_LANG) or {
    "title": "❓ Ask",
    "caption": "",
    "input_label": "Question",
    "placeholder": "...",
    "slider": "Top K",
    "btn_ask": "Ask",
    "thinking": "Thinking...",
    "confidence_label": "Confidence",
    "cost_label": "Cost",
    "sources": "Sources",
    "no_sources": "No sources used.",
    "btn_sidebar": "❓ Ask",
}

_WIKI_LINK_RE = re.compile(r"\[\[([^\]]+)\]\]")

# Log entry header: ## 2026-05-12T10:00:01Z — filename.pdf
_LOG_HEADER_RE = re.compile(
    r"^## (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z) — (.+)$"
)
# Individual fields inside a log entry
_LOG_FIELD_RE = re.compile(r"^\s*-\s*\*\*(.+?)\*\*:\s*(.+)$")

# ---------------------------------------------------------------------------
# Page config (must be the very first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(page_title="LLM Wiki", page_icon="📚", layout="wide")

# ---------------------------------------------------------------------------
# Deep-link: sync URL query params → session_state on every rerun.
# This runs before any rendering so that a fresh page load with
# ?nav=page&slug=transformers opens that wiki page immediately.
# ---------------------------------------------------------------------------
_qp = st.query_params
if "nav" in _qp and st.session_state.get("nav") != _qp["nav"]:
    st.session_state["nav"] = _qp["nav"]
if "slug" in _qp and st.session_state.get("slug") != _qp["slug"]:
    st.session_state["slug"] = _qp["slug"]

# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------


def _nav(page: str, slug: str = "") -> None:
    """Switch to *page*, update session_state, and push the URL query params.

    Writing to ``st.query_params`` makes the browser URL reflect the current
    view so links can be shared and the browser back button works.
    """
    st.session_state["nav"] = page
    st.session_state["slug"] = slug
    st.query_params["nav"] = page
    if slug:
        st.query_params["slug"] = slug
    elif "slug" in st.query_params:
        del st.query_params["slug"]


def _current() -> tuple[str, str]:
    return (
        st.session_state.get("nav", "index"),
        st.session_state.get("slug", ""),
    )


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _wiki_slugs() -> list[str]:
    """Return sorted list of all wiki page stems."""
    if not WIKI_DIR.exists():
        return []
    return sorted(p.stem for p in WIKI_DIR.glob("*.md"))


def _fmt_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")


def _fmt_size(path: Path) -> str:
    b = path.stat().st_size
    return f"{b / 1024:.1f} KB" if b >= 1024 else f"{b} B"


def _parse_log() -> list[dict]:  # type: ignore[type-arg]
    """Parse log.md into a list of entry dicts (newest first)."""
    if not LOG_PATH.exists():
        return []
    text = LOG_PATH.read_text(encoding="utf-8")
    entries: list[dict] = []  # type: ignore[type-arg]
    current: dict | None = None  # type: ignore[type-arg]
    for line in text.splitlines():
        m = _LOG_HEADER_RE.match(line)
        if m:
            if current is not None:
                entries.append(current)
            current = {"timestamp": m.group(1), "filename": m.group(2), "fields": {}}
            continue
        if current is not None:
            fm = _LOG_FIELD_RE.match(line)
            if fm:
                current["fields"][fm.group(1)] = fm.group(2).strip()
    if current is not None:
        entries.append(current)
    entries.reverse()  # newest first
    return entries


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 📚 LLM Wiki")
    st.caption("Read-only viewer")
    st.divider()

    if st.button("📚 Wiki Index", use_container_width=True):
        _nav("index")
        st.rerun()
    if st.button("📋 Changelog", use_container_width=True):
        _nav("changelog")
        st.rerun()
    if st.button("📊 Stats", use_container_width=True):
        _nav("stats")
        st.rerun()
    if st.button(_UI["btn_sidebar"], use_container_width=True):
        _nav("ask")
        st.rerun()

    st.divider()
    slugs_all = _wiki_slugs()
    if slugs_all:
        st.caption(f"Pages ({len(slugs_all)})")
        for _slug in slugs_all:
            if st.button(_slug, key=f"sb_{_slug}", use_container_width=True):
                _nav("page", _slug)
                st.rerun()
    else:
        st.caption("No pages yet")

# ---------------------------------------------------------------------------
# Main area — dispatch to the active page
# ---------------------------------------------------------------------------

nav, slug = _current()


# ── Wiki Index ──────────────────────────────────────────────────────────────
if nav == "index":
    st.title("📚 Wiki Index")

    if not INDEX_PATH.exists():
        st.info(
            "🌱 **Wiki is empty** — upload your first file via the API to get started!\n\n"
            "```bash\ncurl -X POST http://localhost:8000/api/v1/files \\\n"
            "     -F 'file=@document.pdf'\n```"
        )
    else:
        page_count = len(slugs_all)
        mtime = _fmt_mtime(INDEX_PATH)

        c1, c2 = st.columns(2)
        c1.metric("Wiki pages", page_count)
        c2.metric("Index last updated", mtime)
        st.divider()

        if page_count == 0:
            st.info("No wiki pages yet — wait for the pipeline to finish processing.")
        else:
            content = INDEX_PATH.read_text(encoding="utf-8")
            lines = content.splitlines()
            pending: list[str] = []

            def _flush(lines_buf: list[str]) -> None:
                for ln in lines_buf:
                    entry_m = re.match(r"^\s*-\s*\[\[([^\]]+)\]\](.*)", ln)
                    if entry_m:
                        s = entry_m.group(1)
                        rest = entry_m.group(2).strip().lstrip("—").strip()
                        b_col, t_col = st.columns([1, 6])
                        if b_col.button("→", key=f"idx_{s}", help=f"Open '{s}'"):
                            _nav("page", s)
                            st.rerun()
                        t_col.markdown(
                            f"`{s}`" + (f" — {rest}" if rest else ""),
                            unsafe_allow_html=False,
                        )
                    elif ln.strip() and not ln.startswith("<!--") and not ln.startswith(">"):
                        st.markdown(escape_dollars_for_streamlit(ln))

            for line in lines:
                if line.startswith("#"):
                    _flush(pending)
                    pending = []
                    lvl = len(line) - len(line.lstrip("#"))
                    text = line.lstrip("# ").strip()
                    if lvl == 1:
                        continue  # skip "Wiki Index" title — already shown
                    elif lvl == 2:
                        st.subheader(text)
                    else:
                        st.markdown(f"{'#' * lvl} {text}")
                else:
                    pending.append(line)
            _flush(pending)

            # Orphan pages (on disk but not in index)
            linked = set(_WIKI_LINK_RE.findall(content))
            orphans = [s for s in slugs_all if s not in linked]
            if orphans:
                st.divider()
                st.subheader("📎 Unlinked pages")
                st.caption("These pages exist on disk but aren't referenced in index.md yet.")
                for s in orphans:
                    if st.button(f"→ {s}", key=f"orph_{s}"):
                        _nav("page", s)
                        st.rerun()


# ── Wiki Page ────────────────────────────────────────────────────────────────
elif nav == "page":
    if st.button("← Back to Index"):
        _nav("index")
        st.rerun()

    if not slug:
        st.warning("No page selected. Use the sidebar to pick a page.")
    else:
        page_path = WIKI_DIR / f"{slug}.md"
        if not page_path.exists():
            st.error(f"**404** — Page `{slug}` not found in `data/wiki/`.")
            st.caption(
                "The page may still be processing, or the slug may be wrong."
            )
        else:
            content = page_path.read_text(encoding="utf-8")

            # Metadata strip
            c1, c2, c3 = st.columns(3)
            c1.metric("Page", slug)
            c2.metric("Last modified", _fmt_mtime(page_path))
            c3.metric("Size", _fmt_size(page_path))
            st.divider()

            # Render markdown — replace [[links]] with inline code so they're
            # visible; clickable buttons are shown separately below.
            # escape_dollars_for_streamlit prevents currency signs from being
            # parsed as LaTeX math delimiters.
            rendered = _WIKI_LINK_RE.sub(r"`[[\1]]`", content)
            st.markdown(escape_dollars_for_streamlit(rendered))

            # Internal links section
            linked_slugs = list(dict.fromkeys(_WIKI_LINK_RE.findall(content)))
            if linked_slugs:
                st.divider()
                st.caption("🔗 **Internal links** found on this page:")
                cols = st.columns(min(len(linked_slugs), 4))
                for i, ls in enumerate(linked_slugs):
                    if (WIKI_DIR / f"{ls}.md").exists():
                        if cols[i % 4].button(f"→ {ls}", key=f"lnk_{ls}"):
                            _nav("page", ls)
                            st.rerun()
                    else:
                        cols[i % 4].caption(f"⚠ {ls} (missing)")


# ── Changelog ────────────────────────────────────────────────────────────────
elif nav == "changelog":
    st.title("📋 Changelog")

    entries = _parse_log()
    if not entries:
        st.info("No ingestion events yet. Upload a file to populate the log.")
    else:
        st.caption(f"{len(entries)} ingestion event(s) — newest first")
        st.divider()

        # Optional filter
        filter_type = st.radio(
            "Filter",
            ["All", "Created", "Updated"],
            horizontal=True,
            label_visibility="collapsed",
        )

        shown = 0
        for entry in entries:
            fields = entry.get("fields", {})
            created = fields.get("Created", "none")
            updated = fields.get("Updated", "none")
            file_id = fields.get("File ID", "—")
            cost = fields.get("Cost", "—")

            has_created = created not in ("none", "", "—")
            has_updated = updated not in ("none", "", "—")

            if filter_type == "Created" and not has_created:
                continue
            if filter_type == "Updated" and not has_updated:
                continue

            shown += 1
            ts = entry["timestamp"]
            fname = entry["filename"]

            action_label = (
                "🆕 Created" if has_created and not has_updated
                else "✏️ Updated" if has_updated and not has_created
                else "🆕✏️ Mixed"
            )

            with st.expander(f"{action_label}  |  {ts}  |  **{fname}**", expanded=False):
                c1, c2, c3 = st.columns(3)
                c1.markdown(f"**File ID:** `{file_id}`")
                c2.markdown(f"**Cost:** {cost}")
                c3.markdown(f"**Time:** {ts}")

                if has_created:
                    st.markdown(f"**Created pages:** {created}")
                if has_updated:
                    st.markdown(f"**Updated pages:** {updated}")

                # Quick-jump buttons for affected pages
                affected = []
                if has_created:
                    affected += [s.strip() for s in created.split(",")]
                if has_updated:
                    affected += [s.strip() for s in updated.split(",")]
                affected = [s for s in affected if s and s != "none"]

                if affected:
                    btn_cols = st.columns(min(len(affected), 5))
                    for i, s in enumerate(affected):
                        if btn_cols[i % 5].button(f"→ {s}", key=f"cl_{ts}_{s}"):
                            _nav("page", s)
                            st.rerun()

        if shown == 0:
            st.info("No entries match the current filter.")


# ── Stats ─────────────────────────────────────────────────────────────────
elif nav == "stats":
    st.title("📊 Stats")

    wiki_pages = _wiki_slugs()
    raw_files = sorted(RAW_DIR.glob("*")) if RAW_DIR.exists() else []

    wiki_size_bytes = (
        sum(p.stat().st_size for p in WIKI_DIR.glob("*.md")) if WIKI_DIR.exists() else 0
    )

    # Last ingestion from log
    entries = _parse_log()
    last_ingest = entries[0]["timestamp"] if entries else "—"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Wiki pages", len(wiki_pages))
    c2.metric("Raw files", len(raw_files))
    c3.metric("Wiki size", f"{wiki_size_bytes / 1024:.1f} KB" if wiki_size_bytes else "0 B")
    c4.metric("Last ingestion", last_ingest)

    if wiki_pages:
        st.divider()
        st.subheader("Page sizes")
        data = []
        for s in wiki_pages:
            p = WIKI_DIR / f"{s}.md"
            data.append(
                {
                    "slug": s,
                    "size_kb": round(p.stat().st_size / 1024, 2),
                    "modified": _fmt_mtime(p),
                }
            )
        data.sort(key=lambda x: x["size_kb"], reverse=True)

        col_hdr1, col_hdr2, col_hdr3 = st.columns([3, 1, 2])
        col_hdr1.markdown("**Slug**")
        col_hdr2.markdown("**Size (KB)**")
        col_hdr3.markdown("**Modified**")
        st.divider()
        for row in data:
            c1, c2, c3 = st.columns([3, 1, 2])
            if c1.button(row["slug"], key=f"stat_{row['slug']}"):
                _nav("page", row["slug"])
                st.rerun()
            c2.markdown(str(row["size_kb"]))
            c3.markdown(row["modified"])

    if entries:
        st.divider()
        st.subheader("Cost summary")
        total_cost = 0.0
        for e in entries:
            cost_str = e["fields"].get("Cost", "$0").lstrip("$")
            try:
                total_cost += float(cost_str)
            except ValueError:
                pass
        c1, c2 = st.columns(2)
        c1.metric("Total ingestion events", len(entries))
        c2.metric("Total LLM cost", f"${total_cost:.4f}")


# ── Ask ──────────────────────────────────────────────────────────────────────
elif nav == "ask":
    st.title(_UI["title"])
    st.caption(_UI["caption"])

    question = st.text_input(
        _UI["input_label"],
        key="ask_question",
        placeholder=_UI["placeholder"],
    )
    top_k = st.slider(_UI["slider"], 1, 10, 5)

    if st.button(_UI["btn_ask"], type="primary", disabled=not question.strip()):
        with st.spinner(_UI["thinking"]):
            data: dict | None = None
            try:
                resp = httpx.post(
                    f"{API_BASE}/api/v1/ask",
                    json={"question": question.strip(), "top_k": top_k},
                    # connect/write timeouts are short; read is generous because
                    # AnswerAgent may embed + call LLM (up to 60 s per attempt,
                    # up to 3 retries on 429/5xx) before it returns.
                    timeout=httpx.Timeout(connect=5.0, read=90.0, write=10.0, pool=5.0),
                )
                resp.raise_for_status()
                data = resp.json()
            except httpx.ReadTimeout:
                st.error(
                    "Сервер не успел ответить за 90 секунд. "
                    "Попробуйте уменьшить top_k или повторите запрос."
                )
                st.stop()
            except httpx.HTTPStatusError as exc:
                st.error(
                    f"Ошибка API ({exc.response.status_code}): "
                    f"{exc.response.text[:300]}"
                )
                st.stop()
            except httpx.RequestError as exc:
                st.error(f"Не удалось подключиться к API: {exc}")
                st.stop()

        if data:
            confidence_badges: dict[str, str] = {
                "high": "🟢 High",
                "medium": "🟡 Medium",
                "low": "🔴 Low",
            }
            badge = confidence_badges.get(data["confidence"], data["confidence"])
            st.markdown(
                f"**{_UI['confidence_label']}:** {badge}"
                f"  ·  **{_UI['cost_label']}:** ${data['cost_usd']:.4f}"
            )
            st.divider()
            st.markdown(escape_dollars_for_streamlit(data["answer"]))

            if data["sources"]:
                st.divider()
                st.subheader(_UI["sources"])
                for src in data["sources"]:
                    cols = st.columns([3, 1])
                    if cols[0].button(
                        f"→ {src['title']} ([[{src['slug']}]])",
                        key=f"src_{src['slug']}",
                    ):
                        _nav("page", src["slug"])
                        st.rerun()
                    cols[1].markdown(f"similarity: `{src['similarity']:.2f}`")
            else:
                st.info(_UI["no_sources"])
