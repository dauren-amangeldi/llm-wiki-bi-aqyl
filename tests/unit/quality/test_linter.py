"""Unit tests for quality/linter.py (LW-14).

All tests use an in-memory wiki (dict[slug, content]) to keep them fast and
dependency-free.  The Linter is a pure function: no mocking required.
"""

from __future__ import annotations

import pytest

from llm_wiki.quality.linter import run_linter
from llm_wiki.quality.models import IssueKind


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CLEAN_WIKI: dict[str, str] = {
    "alpha": "# Alpha\nSee [[beta]] and [[gamma]].",
    "beta": "# Beta\nSee [[alpha]] and [[gamma]].\n\n## Backlinks\n- [[alpha]]",
    "gamma": "# Gamma\nRefers to [[alpha]].\n\n## Backlinks\n- [[alpha]]\n- [[beta]]",
}

CLEAN_ROOT_SECTIONS: set[str] = {"general"}

CURRENT_YEAR = 2026


# ---------------------------------------------------------------------------
# Dead link tests
# ---------------------------------------------------------------------------

class TestDeadLinks:
    def test_dead_link_detected(self) -> None:
        wiki = {
            "alpha": "# Alpha\nSee [[ghost]].",
            "beta": "# Beta\nSee [[alpha]].",
        }
        issues = run_linter(wiki, set(), CURRENT_YEAR)
        dead = [i for i in issues if i.kind == IssueKind.DEAD_LINK]
        assert len(dead) == 1
        assert dead[0].page_slug == "alpha"
        assert dead[0].related_slugs == ("ghost",)

    def test_no_dead_link_for_self_reference(self) -> None:
        wiki = {"self-page": "# Self\nSee [[self-page]]."}
        issues = run_linter(wiki, set(), CURRENT_YEAR)
        dead = [i for i in issues if i.kind == IssueKind.DEAD_LINK]
        assert dead == []

    def test_multiple_dead_links_on_one_page(self) -> None:
        wiki = {
            "page": "# Page\nLinks: [[missing-one]], [[missing-two]], [[missing-three]]."
        }
        issues = run_linter(wiki, set(), CURRENT_YEAR)
        dead = [i for i in issues if i.kind == IssueKind.DEAD_LINK]
        assert len(dead) == 3
        targets = {i.related_slugs[0] for i in dead}
        assert targets == {"missing-one", "missing-two", "missing-three"}

    def test_no_dead_links_in_clean_wiki(self) -> None:
        issues = run_linter(CLEAN_WIKI, CLEAN_ROOT_SECTIONS, CURRENT_YEAR)
        dead = [i for i in issues if i.kind == IssueKind.DEAD_LINK]
        assert dead == []


# ---------------------------------------------------------------------------
# Orphan page tests
# ---------------------------------------------------------------------------

class TestOrphanPages:
    def test_orphan_detection(self) -> None:
        wiki = {
            "popular": "# Popular\nLots of content.",
            "lonely": "# Lonely\nNo one links here.",
        }
        # popular links to no one; lonely links to no one — both orphans
        issues = run_linter(wiki, set(), CURRENT_YEAR)
        orphans = {i.page_slug for i in issues if i.kind == IssueKind.ORPHAN_PAGE}
        assert "lonely" in orphans

    def test_root_section_not_orphan(self) -> None:
        wiki = {
            "root-section": "# Root\nTop-level section.",
            "child": "# Child\nLinks to [[root-section]].",
        }
        issues = run_linter(wiki, {"root-section"}, CURRENT_YEAR)
        orphans = {i.page_slug for i in issues if i.kind == IssueKind.ORPHAN_PAGE}
        assert "root-section" not in orphans

    def test_page_with_incoming_link_not_orphan(self) -> None:
        wiki = {
            "page-a": "# A\nSee [[page-b]].",
            "page-b": "# B\nContent.",
        }
        issues = run_linter(wiki, set(), CURRENT_YEAR)
        orphans = {i.page_slug for i in issues if i.kind == IssueKind.ORPHAN_PAGE}
        assert "page-b" not in orphans

    def test_no_orphans_in_clean_wiki(self) -> None:
        # Every page in CLEAN_WIKI is linked by at least one other page
        issues = run_linter(CLEAN_WIKI, CLEAN_ROOT_SECTIONS, CURRENT_YEAR)
        orphans = [i for i in issues if i.kind == IssueKind.ORPHAN_PAGE]
        assert orphans == []


# ---------------------------------------------------------------------------
# Stale date tests
# ---------------------------------------------------------------------------

class TestStaleDates:
    def test_stale_date_detected_russian(self) -> None:
        wiki = {"page": "В 2023 году вышел новый релиз."}
        issues = run_linter(wiki, set(), 2026)
        stale = [i for i in issues if i.kind == IssueKind.STALE_DATE]
        assert len(stale) == 1
        assert stale[0].page_slug == "page"
        assert "2023" in stale[0].description

    def test_stale_date_detected_english(self) -> None:
        wiki = {"page": "The framework was released in 2023 and gained popularity."}
        issues = run_linter(wiki, set(), 2026)
        stale = [i for i in issues if i.kind == IssueKind.STALE_DATE]
        assert len(stale) == 1
        assert "2023" in stale[0].description

    def test_stale_date_not_triggered_for_last_year(self) -> None:
        """One-year-old references (current_year - 1) should NOT be flagged."""
        wiki = {"page": "В 2025 году вышел новый релиз."}
        issues = run_linter(wiki, set(), 2026)
        stale = [i for i in issues if i.kind == IssueKind.STALE_DATE]
        assert stale == []

    def test_stale_date_not_triggered_for_current_year(self) -> None:
        wiki = {"page": "In 2026 the project launched."}
        issues = run_linter(wiki, set(), 2026)
        stale = [i for i in issues if i.kind == IssueKind.STALE_DATE]
        assert stale == []

    def test_stale_date_boundary_exactly_two_years_ago(self) -> None:
        """Year current_year - 2 should be flagged (< current_year - 1)."""
        wiki = {"page": "In 2024 this was the latest version."}
        issues = run_linter(wiki, set(), 2026)
        stale = [i for i in issues if i.kind == IssueKind.STALE_DATE]
        assert len(stale) == 1

    def test_multiple_stale_years_one_page(self) -> None:
        wiki = {"page": "In 2020, in 2021, and in 2022 major events occurred."}
        issues = run_linter(wiki, set(), 2026)
        stale = [i for i in issues if i.kind == IssueKind.STALE_DATE]
        assert len(stale) == 3
        years = {int(w) for i in stale for w in i.description.split() if w.isdigit() and len(w) == 4}
        assert {2020, 2021, 2022}.issubset(years)


# ---------------------------------------------------------------------------
# Combined / idempotency tests
# ---------------------------------------------------------------------------

class TestCombined:
    def test_clean_wiki_returns_empty(self) -> None:
        issues = run_linter(CLEAN_WIKI, CLEAN_ROOT_SECTIONS, CURRENT_YEAR)
        assert issues == []

    def test_idempotency(self) -> None:
        wiki = {
            "alpha": "# Alpha\nSee [[ghost]] in 2022.",
            "beta": "# Beta\nSee [[alpha]].",
        }
        run1 = run_linter(wiki, set(), 2026)
        run2 = run_linter(wiki, set(), 2026)
        assert run1 == run2

    def test_empty_wiki_returns_empty(self) -> None:
        assert run_linter({}, set(), CURRENT_YEAR) == []

    def test_dead_link_fixture(self) -> None:
        """Verify the dead_links fixture has the expected dead links."""
        import glob
        from pathlib import Path

        fixture_dir = (
            Path(__file__).parent.parent.parent
            / "fixtures"
            / "sample_wikis"
            / "dead_links"
        )
        wiki: dict[str, str] = {
            p.stem: p.read_text(encoding="utf-8")
            for p in fixture_dir.glob("*.md")
            if p.name != "index.md"
        }
        issues = run_linter(wiki, set(), CURRENT_YEAR)
        dead = [i for i in issues if i.kind == IssueKind.DEAD_LINK]
        # python links to data-science, machine-learning; rust links to cpp;
        # golang links to microservices, cloud-native — at least 3 dead links
        assert len(dead) >= 3
