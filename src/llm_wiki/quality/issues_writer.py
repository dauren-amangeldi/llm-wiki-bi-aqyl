"""Atomic reader/writer for ``data/issues.md``.

``issues.md`` is structured as two independent sections separated by an HTML
comment sentinel (so section boundaries survive normal Markdown editing):

    # Wiki Quality Issues

    <!-- section:auto-detected -->
    ## 🔎 Auto-detected Issues
    ...
    <!-- /section:auto-detected -->

    <!-- section:llm-flagged -->
    ## 🤖 LLM-flagged Issues
    ...
    <!-- /section:llm-flagged -->

``upsert_section`` replaces everything between the sentinel comments while
leaving the other section untouched.  The file is written atomically via
``storage.filesystem.atomic_write`` so a crash mid-write leaves a consistent
previous state on disk.

Design notes:
- **Snapshot, not append-only.**  Every call to ``upsert_section`` is a
  complete fresh snapshot of that section.  Old findings vanish — if the
  human wants history, ``git log issues.md`` is the right tool.
- **Not idempotent on content** (same issues → same text) but idempotent on
  structure (no duplicate sections accumulate).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import structlog

from llm_wiki.quality.models import Issue, IssueSection
from llm_wiki.storage.filesystem import atomic_write

logger = structlog.get_logger(__name__)

# Section-sentinel comment pattern
_SENTINEL_START = "<!-- section:{key} -->"
_SENTINEL_END = "<!-- /section:{key} -->"

_SECTION_RE_TEMPLATE = (
    r"<!-- section:{key} -->.*?<!-- /section:{key} -->"
)

_SECTION_TITLES: dict[IssueSection, str] = {
    IssueSection.AUTO_DETECTED: "## 🔎 Auto-detected Issues",
    IssueSection.LLM_FLAGGED: "## 🤖 LLM-flagged Issues",
}

_ISSUES_HEADER = "# Wiki Quality Issues\n"

_EMPTY_PLACEHOLDER: dict[IssueSection, str] = {
    IssueSection.AUTO_DETECTED: "_No auto-detected issues._",
    IssueSection.LLM_FLAGGED: "_No LLM-flagged issues._",
}


def _render_section(section: IssueSection, issues: list[Issue]) -> str:
    """Return the full sentinel-wrapped Markdown block for *section*.

    Args:
        section: Which quality section this is.
        issues: Issues to include.  May be empty.

    Returns:
        Markdown string starting and ending with sentinel comments.
    """
    key = str(section)
    title = _SECTION_TITLES[section]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = [
        _SENTINEL_START.format(key=key),
        title,
        f"_Last updated: {now}_",
        "",
    ]

    if not issues:
        lines.append(_EMPTY_PLACEHOLDER[section])
    else:
        for issue in sorted(issues, key=lambda i: (i.kind, i.page_slug)):
            related = (
                ", ".join(f"`[[{s}]]`" for s in issue.related_slugs)
                if issue.related_slugs
                else ""
            )
            slug_md = f"**`[[{issue.page_slug}]]`**"
            desc = issue.description
            if related:
                lines.append(f"- {slug_md} [{issue.kind}] {desc} → {related}")
            else:
                lines.append(f"- {slug_md} [{issue.kind}] {desc}")

    lines.append("")
    lines.append(_SENTINEL_END.format(key=key))
    return "\n".join(lines)


def _stamp_issues(issues: list[Issue]) -> list[Issue]:
    """Return *issues* with ``detected_at`` filled in (current UTC time)."""
    now = datetime.now(timezone.utc)
    return [
        Issue(
            kind=i.kind,
            section=i.section,
            page_slug=i.page_slug,
            description=i.description,
            related_slugs=i.related_slugs,
            detected_at=now,
        )
        for i in issues
    ]


def upsert_section(
    issues_path: Path,
    section: IssueSection,
    issues: list[Issue],
) -> None:
    """Replace *section* in *issues_path* with the rendered *issues*.

    If the file does not exist it is created with both sections (the
    untouched section shows the empty placeholder text).  The file is
    written atomically; a crash mid-write leaves the previous version intact.

    Args:
        issues_path: Path to ``data/issues.md``.
        section: Which section to replace (auto-detected or llm-flagged).
        issues: Fresh issue snapshot.  Pass ``[]`` to clear the section.
    """
    stamped = _stamp_issues(issues)
    new_block = _render_section(section, stamped)
    key = str(section)
    pattern = re.compile(
        _SECTION_RE_TEMPLATE.format(key=re.escape(key)),
        re.DOTALL,
    )

    if issues_path.exists():
        content = issues_path.read_text(encoding="utf-8")
    else:
        # Bootstrap file with both sections
        content = _bootstrap_issues_md()

    if pattern.search(content):
        new_content = pattern.sub(new_block, content)
    else:
        # Section sentinels missing — append the section
        if not content.endswith("\n\n"):
            content = content.rstrip("\n") + "\n\n"
        new_content = content + new_block + "\n"

    atomic_write(issues_path, new_content)
    logger.info(
        "issues_section_updated",
        section=str(section),
        issue_count=len(issues),
    )


def read_section(issues_path: Path, section: IssueSection) -> str:
    """Return the raw Markdown content inside *section*'s sentinel block.

    Args:
        issues_path: Path to ``data/issues.md``.
        section: Which section to extract.

    Returns:
        Content string, or empty string if file or section is absent.
    """
    if not issues_path.exists():
        return ""
    key = str(section)
    pattern = re.compile(
        _SECTION_RE_TEMPLATE.format(key=re.escape(key)),
        re.DOTALL,
    )
    content = issues_path.read_text(encoding="utf-8")
    m = pattern.search(content)
    return m.group(0) if m else ""


def _bootstrap_issues_md() -> str:
    """Return the initial ``issues.md`` content with both empty sections."""
    parts = [_ISSUES_HEADER, ""]
    for section in IssueSection:
        parts.append(_render_section(section, []))
        parts.append("")
    return "\n".join(parts)
