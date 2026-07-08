"""Unit tests for quality/issues_writer.py (Postgres-backed issues_report)."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_wiki.quality.issues_writer import read_section, upsert_section
from llm_wiki.quality.models import Issue, IssueKind, IssueSection

# Dummy path — issues_writer ignores it now (content lives in Postgres), but the
# public signature still accepts it for backwards compatibility.
_DUMMY = Path("/tmp/ignored-issues.md")


@pytest.fixture(autouse=True)
def _issues_db(db_engine):  # type: ignore[no-untyped-def]  # noqa: ANN001
    """issues_writer reads/writes the issues_report table — clean DB per test."""
    yield


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


class TestUpsertSection:
    def test_empty_issues_writes_placeholder(self) -> None:
        upsert_section(_DUMMY, IssueSection.AUTO_DETECTED, [])
        content = read_section(_DUMMY, IssueSection.AUTO_DETECTED)
        assert "_No auto-detected issues._" in content

    def test_issues_appear_in_section(self) -> None:
        issues = [_make_issue(page_slug="foo", description="Dead link to bar.")]
        upsert_section(_DUMMY, IssueSection.AUTO_DETECTED, issues)
        content = read_section(_DUMMY, IssueSection.AUTO_DETECTED)
        assert "foo" in content
        assert "Dead link to bar." in content
        assert "dead_link" in content

    def test_upsert_replaces_section_not_appends(self) -> None:
        upsert_section(_DUMMY, IssueSection.AUTO_DETECTED, [_make_issue(page_slug="first")])
        upsert_section(_DUMMY, IssueSection.AUTO_DETECTED, [_make_issue(page_slug="second")])
        content = read_section(_DUMMY, IssueSection.AUTO_DETECTED)
        assert "first" not in content
        assert "second" in content

    def test_other_section_preserved(self) -> None:
        upsert_section(
            _DUMMY,
            IssueSection.LLM_FLAGGED,
            [_make_issue(kind=IssueKind.CONTRADICTION, page_slug="llm-page", section=IssueSection.LLM_FLAGGED)],
        )
        upsert_section(_DUMMY, IssueSection.AUTO_DETECTED, [_make_issue(page_slug="linter-page")])
        assert "llm-page" in read_section(_DUMMY, IssueSection.LLM_FLAGGED)
        assert "linter-page" in read_section(_DUMMY, IssueSection.AUTO_DETECTED)

    def test_sentinel_appears_exactly_once_per_section(self) -> None:
        upsert_section(_DUMMY, IssueSection.AUTO_DETECTED, [])
        upsert_section(_DUMMY, IssueSection.AUTO_DETECTED, [])
        content = read_section(_DUMMY, IssueSection.AUTO_DETECTED)
        assert content.count("<!-- section:auto-detected -->") == 1
        assert content.count("<!-- /section:auto-detected -->") == 1

    def test_related_slugs_rendered(self) -> None:
        upsert_section(_DUMMY, IssueSection.AUTO_DETECTED, [_make_issue(related_slugs=("target-page",))])
        assert "target-page" in read_section(_DUMMY, IssueSection.AUTO_DETECTED)

    def test_issues_sorted_by_kind_then_slug(self) -> None:
        issues = [
            _make_issue(kind=IssueKind.STALE_DATE, page_slug="z-page"),
            _make_issue(kind=IssueKind.DEAD_LINK, page_slug="a-page"),
            _make_issue(kind=IssueKind.ORPHAN_PAGE, page_slug="m-page"),
        ]
        upsert_section(_DUMMY, IssueSection.AUTO_DETECTED, issues)
        content = read_section(_DUMMY, IssueSection.AUTO_DETECTED)
        assert content.index("a-page") < content.index("m-page") < content.index("z-page")


class TestReadSection:
    def test_read_returns_empty_when_never_written(self) -> None:
        assert read_section(_DUMMY, IssueSection.AUTO_DETECTED) == ""

    def test_read_returns_section_content(self) -> None:
        upsert_section(_DUMMY, IssueSection.AUTO_DETECTED, [_make_issue(page_slug="target-slug")])
        content = read_section(_DUMMY, IssueSection.AUTO_DETECTED)
        assert "target-slug" in content
        assert "<!-- section:auto-detected -->" in content

    def test_read_other_section_absent_returns_empty(self) -> None:
        upsert_section(_DUMMY, IssueSection.AUTO_DETECTED, [])
        assert read_section(_DUMMY, IssueSection.LLM_FLAGGED) == ""
