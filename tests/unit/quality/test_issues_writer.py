"""Unit tests for quality/issues_writer.py (LW-14/15)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from llm_wiki.quality.issues_writer import _bootstrap_issues_md, read_section, upsert_section
from llm_wiki.quality.models import Issue, IssueKind, IssueSection


def _make_issue(
    kind: IssueKind = IssueKind.DEAD_LINK,
    page_slug: str = "alpha",
    description: str = "Test issue.",
    section: IssueSection = IssueSection.AUTO_DETECTED,
    related_slugs: tuple[str, ...] = (),
) -> Issue:
    return Issue(
        kind=kind,
        section=section,
        page_slug=page_slug,
        description=description,
        related_slugs=related_slugs,
    )


class TestBootstrap:
    def test_bootstrap_contains_both_sections(self) -> None:
        md = _bootstrap_issues_md()
        assert "<!-- section:auto-detected -->" in md
        assert "<!-- section:llm-flagged -->" in md
        assert "<!-- /section:auto-detected -->" in md
        assert "<!-- /section:llm-flagged -->" in md

    def test_bootstrap_has_header(self) -> None:
        md = _bootstrap_issues_md()
        assert "# Wiki Quality Issues" in md

    def test_bootstrap_empty_placeholders(self) -> None:
        md = _bootstrap_issues_md()
        assert "_No auto-detected issues._" in md
        assert "_No LLM-flagged issues._" in md


class TestUpsertSection:
    def test_creates_file_when_absent(self, tmp_path: Path) -> None:
        issues_path = tmp_path / "issues.md"
        upsert_section(issues_path, IssueSection.AUTO_DETECTED, [])
        assert issues_path.exists()

    def test_empty_issues_writes_placeholder(self, tmp_path: Path) -> None:
        issues_path = tmp_path / "issues.md"
        upsert_section(issues_path, IssueSection.AUTO_DETECTED, [])
        content = issues_path.read_text()
        assert "_No auto-detected issues._" in content

    def test_issues_appear_in_section(self, tmp_path: Path) -> None:
        issues_path = tmp_path / "issues.md"
        issues = [_make_issue(page_slug="foo", description="Dead link to bar.")]
        upsert_section(issues_path, IssueSection.AUTO_DETECTED, issues)
        content = issues_path.read_text()
        assert "foo" in content
        assert "Dead link to bar." in content
        assert "dead_link" in content

    def test_upsert_replaces_section_not_appends(self, tmp_path: Path) -> None:
        issues_path = tmp_path / "issues.md"
        upsert_section(
            issues_path, IssueSection.AUTO_DETECTED,
            [_make_issue(page_slug="first")]
        )
        upsert_section(
            issues_path, IssueSection.AUTO_DETECTED,
            [_make_issue(page_slug="second")]
        )
        content = issues_path.read_text()
        assert "first" not in content
        assert "second" in content

    def test_other_section_preserved(self, tmp_path: Path) -> None:
        issues_path = tmp_path / "issues.md"
        # Write LLM-flagged section first
        upsert_section(
            issues_path, IssueSection.LLM_FLAGGED,
            [_make_issue(
                kind=IssueKind.CONTRADICTION,
                page_slug="llm-page",
                section=IssueSection.LLM_FLAGGED,
            )]
        )
        # Overwrite auto-detected section
        upsert_section(
            issues_path, IssueSection.AUTO_DETECTED,
            [_make_issue(page_slug="linter-page")]
        )
        content = issues_path.read_text()
        assert "llm-page" in content
        assert "linter-page" in content

    def test_sentinel_appears_exactly_once_per_section(self, tmp_path: Path) -> None:
        issues_path = tmp_path / "issues.md"
        upsert_section(issues_path, IssueSection.AUTO_DETECTED, [])
        upsert_section(issues_path, IssueSection.AUTO_DETECTED, [])
        content = issues_path.read_text()
        assert content.count("<!-- section:auto-detected -->") == 1
        assert content.count("<!-- /section:auto-detected -->") == 1

    def test_related_slugs_rendered(self, tmp_path: Path) -> None:
        issues_path = tmp_path / "issues.md"
        issue = _make_issue(related_slugs=("target-page",))
        upsert_section(issues_path, IssueSection.AUTO_DETECTED, [issue])
        content = issues_path.read_text()
        assert "target-page" in content

    def test_issues_sorted_by_kind_then_slug(self, tmp_path: Path) -> None:
        issues_path = tmp_path / "issues.md"
        issues = [
            _make_issue(kind=IssueKind.STALE_DATE, page_slug="z-page"),
            _make_issue(kind=IssueKind.DEAD_LINK, page_slug="a-page"),
            _make_issue(kind=IssueKind.ORPHAN_PAGE, page_slug="m-page"),
        ]
        upsert_section(issues_path, IssueSection.AUTO_DETECTED, issues)
        content = issues_path.read_text()
        pos_dead = content.index("a-page")
        pos_orphan = content.index("m-page")
        pos_stale = content.index("z-page")
        assert pos_dead < pos_orphan < pos_stale


class TestReadSection:
    def test_read_returns_empty_when_file_absent(self, tmp_path: Path) -> None:
        result = read_section(tmp_path / "nonexistent.md", IssueSection.AUTO_DETECTED)
        assert result == ""

    def test_read_returns_section_content(self, tmp_path: Path) -> None:
        issues_path = tmp_path / "issues.md"
        upsert_section(
            issues_path, IssueSection.AUTO_DETECTED,
            [_make_issue(page_slug="target-slug")]
        )
        content = read_section(issues_path, IssueSection.AUTO_DETECTED)
        assert "target-slug" in content
        assert "<!-- section:auto-detected -->" in content

    def test_read_other_section_absent_returns_empty(self, tmp_path: Path) -> None:
        issues_path = tmp_path / "issues.md"
        # Only write auto-detected; read llm-flagged
        upsert_section(issues_path, IssueSection.AUTO_DETECTED, [])
        # Remove the llm-flagged section manually
        content = issues_path.read_text()
        content = re.sub(
            r"<!-- section:llm-flagged -->.*?<!-- /section:llm-flagged -->",
            "",
            content,
            flags=re.DOTALL,
        )
        issues_path.write_text(content)
        result = read_section(issues_path, IssueSection.LLM_FLAGGED)
        assert result == ""
