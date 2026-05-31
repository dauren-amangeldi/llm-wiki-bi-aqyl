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

import streamlit as st

# ---------------------------------------------------------------------------
# Paths (configurable via environment for Docker flexibility)
# ---------------------------------------------------------------------------

DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
WIKI_DIR = DATA_DIR / "wiki"
INDEX_PATH = DATA_DIR / "index.md"
LOG_PATH = DATA_DIR / "log.md"
RAW_DIR = DATA_DIR / "raw"

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
# Session-state helpers
# ---------------------------------------------------------------------------


def _nav(page: str, slug: str = "") -> None:
    """Switch to *page*, optionally setting the current wiki slug."""
    st.session_state["nav"] = page
    st.session_state["slug"] = slug


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
                        st.markdown(ln)

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
            rendered = _WIKI_LINK_RE.sub(r"`[[\1]]`", content)
            st.markdown(rendered)

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
