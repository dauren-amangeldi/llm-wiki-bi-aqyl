"""Lightweight summary extraction from plain-text documents.

Used by the Search Agent to distil long documents into a compact fragment
that fits within the embedding / LLM context window.
"""

import re

# Patterns for explicit summary sections
_SUMMARY_HEADERS = re.compile(
    r"^#+\s*(tl[;:\s]?dr|summary|abstract|overview|introduction|executive\s+summary)",
    re.IGNORECASE | re.MULTILINE,
)
_NEXT_HEADER = re.compile(r"^#+\s", re.MULTILINE)


def extract_summary(text: str, max_chars: int = 8_000) -> str:
    """Extract a compact summary from *text*.

    Strategy (in priority order):
    1. If the document contains a ``TL;DR``, ``Summary``, or ``Abstract``
       section, extract its content (up to *max_chars*).
    2. Otherwise, return the first *max_chars* characters.

    The result is always stripped and truncated to *max_chars* characters.

    Args:
        text: Plain-text document content.
        max_chars: Maximum character count for the output.  Defaults to
            ~2 000 tokens (at 4 chars/token) which fits comfortably in most
            LLM context windows.

    Returns:
        Compact summary string, at most *max_chars* characters.
    """
    if not text:
        return ""

    # Try to locate an explicit summary section
    match = _SUMMARY_HEADERS.search(text)
    if match:
        section_start = match.end()
        # Find where the next heading begins (marks end of this section)
        next_match = _NEXT_HEADER.search(text, section_start)
        section_end = next_match.start() if next_match else len(text)
        excerpt = text[section_start:section_end].strip()
        if excerpt:
            return excerpt[:max_chars]

    # Fallback: first max_chars characters
    return text[:max_chars].strip()
