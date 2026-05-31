"""Deterministic wiki quality checks — LW-14.

All checks are **pure functions**: no I/O, no global state, no LLM calls.
The orchestrator (or CLI) is responsible for reading pages from disk and
writing the resulting ``Issue`` list to ``issues.md`` via ``issues_writer``.

Three checks are implemented:

1. **Dead links** — ``[[slug]]`` references that have no corresponding file.
2. **Orphan pages** — pages with no incoming links (and not a root section).
3. **Stale dates** — text mentions a year more than one calendar year in
   the past as if it were current.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from llm_wiki.quality.models import Issue, IssueKind, IssueSection
from llm_wiki.utils.backlinks import extract_outgoing_links

# Patterns for stale-date detection (Russian and English)
_STALE_DATE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"в\s+(20\d{2})\s+(?:году|г\.)", re.IGNORECASE),
    re.compile(r"in\s+(20\d{2})\b", re.IGNORECASE),
    re.compile(r"\b(20\d{2})\s+(?:году|г\.)", re.IGNORECASE),
]

_LINTER_SECTION = IssueSection.AUTO_DETECTED


def run_linter(
    wiki_pages: dict[str, str],
    index_root_sections: set[str],
    current_year: int | None = None,
) -> list[Issue]:
    """Run all deterministic quality checks and return found issues.

    This is the **only** public function in this module.  It is a pure
    function: given the same inputs it always returns the same outputs and
    performs no side effects.

    Args:
        wiki_pages: Mapping of ``{slug: markdown_content}`` for every page.
        index_root_sections: Slug names of root sections in ``index.md``
            (e.g. the ``## General`` heading slug).  These are excluded from
            orphan detection — they exist at the top of the hierarchy and
            are expected to have no incoming links.
        current_year: Override the current year (for testing).  Defaults to
            the actual current calendar year.

    Returns:
        Flat list of :class:`Issue` objects, one per finding.  Empty list
        means the wiki is structurally clean.
    """
    if current_year is None:
        current_year = datetime.now(timezone.utc).year

    issues: list[Issue] = []
    issues.extend(_check_dead_links(wiki_pages))
    issues.extend(_check_orphan_pages(wiki_pages, index_root_sections))
    issues.extend(_check_stale_dates(wiki_pages, current_year))
    return issues


# ---------------------------------------------------------------------------
# Individual checkers — each returns list[Issue]
# ---------------------------------------------------------------------------


def _check_dead_links(wiki_pages: dict[str, str]) -> list[Issue]:
    """Return one DEAD_LINK Issue per broken ``[[slug]]`` reference.

    A link is "dead" if the target slug does not exist in *wiki_pages*.
    Self-references (a page linking to itself) are deliberately excluded —
    they are handled by the backlinks-sync layer and are not a Linter concern.

    Args:
        wiki_pages: Slug → content mapping.

    Returns:
        List of DEAD_LINK issues.
    """
    issues: list[Issue] = []
    known = set(wiki_pages)
    for slug, content in sorted(wiki_pages.items()):
        for target in extract_outgoing_links(content):
            if target == slug:
                continue  # self-reference — not a dead link
            if target not in known:
                issues.append(
                    Issue(
                        kind=IssueKind.DEAD_LINK,
                        section=_LINTER_SECTION,
                        page_slug=slug,
                        description=f"References [[{target}]] which does not exist.",
                        related_slugs=(target,),
                    )
                )
    return issues


def _check_orphan_pages(
    wiki_pages: dict[str, str],
    index_root_sections: set[str],
) -> list[Issue]:
    """Return one ORPHAN_PAGE Issue per page with no incoming links.

    A page is an orphan if no other page in *wiki_pages* contains a
    ``[[slug]]`` reference to it, **and** it is not listed in
    *index_root_sections*.

    Args:
        wiki_pages: Slug → content mapping.
        index_root_sections: Slugs that are top-level sections in index.md.

    Returns:
        List of ORPHAN_PAGE issues.
    """
    # Build inverted index: target_slug → set of source slugs
    incoming: dict[str, set[str]] = {slug: set() for slug in wiki_pages}
    for slug, content in wiki_pages.items():
        for target in extract_outgoing_links(content):
            if target in incoming and target != slug:
                incoming[target].add(slug)

    issues: list[Issue] = []
    for slug in sorted(wiki_pages):
        if slug in index_root_sections:
            continue  # root sections are exempt
        if not incoming.get(slug):
            issues.append(
                Issue(
                    kind=IssueKind.ORPHAN_PAGE,
                    section=_LINTER_SECTION,
                    page_slug=slug,
                    description="No other page links to this page.",
                )
            )
    return issues


def _check_stale_dates(
    wiki_pages: dict[str, str],
    current_year: int,
) -> list[Issue]:
    """Return one STALE_DATE Issue per page mentioning a year that is stale.

    A year is considered "stale" if it is at least 2 years before the
    current year (i.e., ``year <= current_year - 2``).  The threshold is
    deliberately conservative: a one-year-old reference may legitimately say
    "last year".  Only years of the form ``20xx`` are matched.

    Note: This check only *flags* potential staleness.  It does not assert
    the content is wrong — historical contexts are valid.

    Args:
        wiki_pages: Slug → content mapping.
        current_year: The year to treat as "now".

    Returns:
        List of STALE_DATE issues, at most one per (page, year) pair.
    """
    issues: list[Issue] = []
    stale_threshold = current_year - 1  # year < current_year - 1 is stale

    for slug in sorted(wiki_pages):
        content = wiki_pages[slug]
        found_years: set[int] = set()
        for pattern in _STALE_DATE_PATTERNS:
            for m in pattern.finditer(content):
                try:
                    year = int(m.group(1))
                except (IndexError, ValueError):
                    continue
                if year < stale_threshold:
                    found_years.add(year)

        for year in sorted(found_years):
            issues.append(
                Issue(
                    kind=IssueKind.STALE_DATE,
                    section=_LINTER_SECTION,
                    page_slug=slug,
                    description=(
                        f"Page references {year} as if current; "
                        f"current year is {current_year}."
                    ),
                )
            )
    return issues
