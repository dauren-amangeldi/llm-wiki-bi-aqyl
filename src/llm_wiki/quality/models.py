"""Shared data models for the wiki quality-control system (LW-14 / LW-15).

Design principles:
- ``Issue`` is frozen so it can be hashed, put in sets, and safely passed
  between threads / asyncio tasks.
- ``IssueSection`` maps each agent type to its own section in ``issues.md``
  so a human reader can instantly see what was found algorithmically vs
  what the LLM flagged.
- ``IssueKind`` is a StrEnum so it serialises cleanly to JSON without extra
  wrappers (plain ``str`` value in the JSON output).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class IssueKind(StrEnum):
    """Categories of quality issues the system can detect."""

    # ── Deterministic / Linter (LW-14) ────────────────────────────────────
    DEAD_LINK = "dead_link"
    """Outgoing ``[[slug]]`` reference to a page that does not exist."""

    ORPHAN_PAGE = "orphan_page"
    """Page that no other page links to (and is not a root index section)."""

    STALE_DATE = "stale_date"
    """Text mentions a year more than one calendar year in the past."""

    # ── LLM / Auditor (LW-15) ─────────────────────────────────────────────
    CONTRADICTION = "contradiction"
    """Two topically-related pages assert conflicting facts."""

    DUPLICATE = "duplicate"
    """Two pages describe the same concept and should be merged."""

    SUSPECTED_STALE = "suspected_stale"
    """Content appears semantically outdated (no regex — by LLM reasoning)."""


class IssueSection(StrEnum):
    """Which section of ``issues.md`` an issue belongs to."""

    AUTO_DETECTED = "auto-detected"
    """Section header written by the deterministic Linter."""

    LLM_FLAGGED = "llm-flagged"
    """Section header written by the LLM Auditor."""


@dataclass(frozen=True)
class Issue:
    """A single quality finding.

    ``Issue`` is immutable (``frozen=True``) so it can be hashed and used
    in sets, enabling idempotent deduplication before writing to ``issues.md``.

    Attributes:
        kind: What kind of problem was detected.
        section: Which section of issues.md this issue belongs to.
        page_slug: The page where the issue was found.
        description: Human-readable explanation of the finding.
        related_slugs: Other pages involved (e.g., the target of a dead link,
            or the duplicate/contradicting page slug).
        detected_at: Wall-clock timestamp set by ``issues_writer`` at write
            time.  ``None`` until persisted.
    """

    kind: IssueKind
    section: IssueSection
    page_slug: str
    description: str
    related_slugs: tuple[str, ...] = field(default_factory=tuple)  # type: ignore[assignment]
    detected_at: datetime | None = field(default=None, compare=False, hash=False)
