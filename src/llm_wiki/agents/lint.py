"""Lint Agent — weekly consistency checker for the wiki.

Checks: dead links, contradictions, duplicates, orphans, stale dates.
NEVER auto-fixes — writes a report to issues.md for human review.
Implemented in LW-14 (rule-based) and LW-15 (LLM-based).
"""

from dataclasses import dataclass, field
from enum import StrEnum

from llm_wiki.agents.base import BaseAgent


class IssueKind(StrEnum):
    """Categories of issues the Lint Agent can detect."""

    DEAD_LINK = "dead_link"
    ORPHAN_PAGE = "orphan_page"
    DUPLICATE = "duplicate"
    CONTRADICTION = "contradiction"
    STALE_DATE = "stale_date"


@dataclass
class LintIssue:
    """A single issue found during a lint run."""

    kind: IssueKind
    page_slug: str
    description: str
    related_slugs: list[str] = field(default_factory=list)


class LintAgent(BaseAgent):
    """Detects quality issues across all wiki pages.

    Runs on a weekly Celery Beat schedule (Monday 03:00 UTC).
    In production uses the Batch API for -50% cost.
    """

    BATCH_SIZE = 50

    async def run(  # type: ignore[override]
        self,
        wiki_pages: list[tuple[str, str]],
    ) -> list[LintIssue]:
        """Scan all wiki pages and return a list of issues.

        Args:
            wiki_pages: List of (slug, markdown_content) tuples.

        Returns:
            All detected issues. Empty list means the wiki is clean.
        """
        raise NotImplementedError("Implemented in LW-14 / LW-15")
