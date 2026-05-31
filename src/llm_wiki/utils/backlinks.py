"""Backlink management utilities for [[wiki-link]] references.

Backlinks are bidirectional: when page A links to page B, B's ## Backlinks
section records A as a backlink.  This module provides three primitives:

  - ``extract_outgoing_links`` — find all [[slug]] in a page body (already done)
  - ``inject_backlink``        — add a slug to the ## Backlinks section
  - ``remove_backlink``        — remove a slug from the ## Backlinks section
  - ``extract_backlinks``      — read slugs listed in ## Backlinks (for the API)

Parsing strategy:
    Line-by-line (no AST) for simplicity and speed.  The ## Backlinks section
    is located by exact header match and bounded by the next heading or EOF.
    Bullet lines are identified by ``_BULLET_RE``; other lines in the section
    are preserved as-is during writes (LLM might add prose, we tolerate it).
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Compiled regexps
# ---------------------------------------------------------------------------

# Matches any [[slug]] in a page body — slug must be kebab-case lowercase.
_WIKI_LINK_RE = re.compile(r"\[\[([a-z0-9][a-z0-9-]*[a-z0-9])\]\]")

# Exact match for the ## Backlinks section header (trailing whitespace allowed).
_BACKLINKS_HEADER_RE = re.compile(r"^##\s+Backlinks\s*$")

# Matches the start of ANY heading — used to detect where the section ends.
_HEADING_START_RE = re.compile(r"^#{1,6}\s")

# Matches a valid backlink bullet: "- [[slug]]" optionally surrounded by whitespace.
_BULLET_RE = re.compile(r"^\s*-\s*\[\[([a-z0-9][a-z0-9-]*[a-z0-9])\]\]\s*$")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _find_backlinks_section(lines: list[str]) -> tuple[int, int] | None:
    """Return ``(start, end)`` line indices of the ``## Backlinks`` section.

    ``start`` is the index of the ``## Backlinks`` header line.
    ``end`` is one past the last line of the section (exclusive upper bound).

    Args:
        lines: Lines of a wiki page, *with* line endings (``splitlines(keepends=True)``).

    Returns:
        ``(start, end)`` tuple, or ``None`` if no ``## Backlinks`` section exists.
    """
    start: int | None = None
    for i, line in enumerate(lines):
        if _BACKLINKS_HEADER_RE.match(line.rstrip("\n")):
            start = i
            break
    if start is None:
        return None
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if _HEADING_START_RE.match(lines[i]):
            end = i
            break
    return start, end


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_outgoing_links(content: str) -> list[str]:
    """Return all ``[[slug]]`` targets referenced in *content*.

    Deduplicates while preserving first-occurrence order.  Does NOT
    filter out the page's own slug — that is the caller's responsibility.

    Args:
        content: Markdown page content.

    Returns:
        Deduplicated list of referenced page slugs, in first-occurrence order.
    """
    return list(dict.fromkeys(_WIKI_LINK_RE.findall(content)))


def inject_backlink(content: str, source_slug: str) -> str:
    """Return *content* with *source_slug* added to the ``## Backlinks`` section.

    - If the section does not exist: appends it at the end of the file as::

        \\n## Backlinks\\n\\n- [[source_slug]]\\n

    - If the section exists: inserts ``- [[source_slug]]`` as a bullet,
      keeping all bullets sorted alphabetically by slug.
    - Idempotent: if ``[[source_slug]]`` is already listed, returns *content*
      unchanged.
    - Non-bullet lines inside the section (e.g., LLM-added prose) are
      preserved; only new bullet lines are inserted.

    Args:
        content: Full Markdown content of the target wiki page.
        source_slug: Slug of the page that links to this page.

    Returns:
        Updated Markdown content, or *content* unchanged if already present.
    """
    lines = content.splitlines(keepends=True)
    section = _find_backlinks_section(lines)

    if section is None:
        # ----------------------------------------------------------------
        # No section: append it after ensuring a blank-line separator.
        # ----------------------------------------------------------------
        if not content:
            return f"## Backlinks\n\n- [[{source_slug}]]\n"
        if content.endswith("\n\n"):
            sep = ""
        elif content.endswith("\n"):
            sep = "\n"
        else:
            sep = "\n\n"
        return content + sep + f"## Backlinks\n\n- [[{source_slug}]]\n"

    start, end = section

    # ----------------------------------------------------------------
    # Section exists: check idempotency and find sorted insert position.
    # ----------------------------------------------------------------
    bullet_positions: list[tuple[int, str]] = []  # (line_index, slug)
    for i in range(start + 1, end):
        m = _BULLET_RE.match(lines[i])
        if m:
            slug = m.group(1)
            if slug == source_slug:
                return content  # already present — idempotent
            bullet_positions.append((i, slug))

    new_bullet = f"- [[{source_slug}]]\n"

    # Find where to insert: first existing bullet with slug > source_slug
    insert_before: int | None = None
    for line_idx, slug in bullet_positions:
        if slug > source_slug:
            insert_before = line_idx
            break

    if insert_before is not None:
        lines.insert(insert_before, new_bullet)
    elif bullet_positions:
        # Append after the last bullet
        last_idx = bullet_positions[-1][0]
        lines.insert(last_idx + 1, new_bullet)
    else:
        # No bullets yet: insert at section start, skipping blank lines
        pos = start + 1
        while pos < end and not lines[pos].strip():
            pos += 1
        lines.insert(pos, new_bullet)

    return "".join(lines)


def remove_backlink(content: str, source_slug: str) -> str:
    """Remove the ``- [[source_slug]]`` bullet from the ``## Backlinks`` section.

    - No-op if the section or bullet is absent.
    - If the section becomes empty (no remaining non-blank lines) after
      removal, the entire ``## Backlinks`` header is also removed, along
      with the preceding blank-line separator, so no dangling empty section
      is left behind.
    - Non-bullet lines in the section are untouched.

    Args:
        content: Full Markdown content of the target wiki page.
        source_slug: Slug of the page whose bullet should be removed.

    Returns:
        Updated Markdown content, or *content* unchanged if nothing matched.
    """
    lines = content.splitlines(keepends=True)
    section = _find_backlinks_section(lines)

    if section is None:
        return content  # nothing to remove

    start, end = section

    # Locate the bullet line for source_slug
    bullet_idx: int | None = None
    for i in range(start + 1, end):
        m = _BULLET_RE.match(lines[i])
        if m and m.group(1) == source_slug:
            bullet_idx = i
            break

    if bullet_idx is None:
        return content  # idempotent: slug not in section

    # Remove the bullet
    del lines[bullet_idx]

    # Re-find section bounds after deletion (indices shifted)
    new_section = _find_backlinks_section(lines)
    if new_section is None:
        return "".join(lines)

    new_start, new_end = new_section

    # Check whether the section now has any non-blank content
    has_content = any(lines[i].strip() for i in range(new_start + 1, new_end))

    if not has_content:
        # Remove the empty section: header + all lines up to new_end
        remove_from = new_start
        # Also eat the blank separator line immediately before the header
        if remove_from > 0 and not lines[remove_from - 1].strip():
            remove_from -= 1
        del lines[remove_from:new_end]

    return "".join(lines)


def extract_backlinks(content: str) -> list[str]:
    """Return slugs listed under the ``## Backlinks`` section, in file order.

    Only bullet lines matching ``- [[slug]]`` are parsed; non-bullet prose
    inserted by the LLM is silently ignored.  Used by ``GET /wiki/{slug}``
    (LW-16) and by ``backlinks_sync`` to compute set differences.

    Args:
        content: Full Markdown content of the target wiki page.

    Returns:
        Ordered list of slug strings.  Empty list if the section is absent.
    """
    lines = content.splitlines(keepends=True)
    section = _find_backlinks_section(lines)
    if section is None:
        return []
    start, end = section
    return [
        m.group(1)
        for i in range(start + 1, end)
        if (m := _BULLET_RE.match(lines[i]))
    ]
