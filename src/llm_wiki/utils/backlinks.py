"""Backlink management utilities for [[wiki-link]] references.

Backlinks are bidirectional: when page A links to page B, B's metadata
records A as a backlink. This module provides helpers for parsing and
updating those references. Implemented in LW-13.
"""

import re
from pathlib import Path


_WIKI_LINK_RE = re.compile(r"\[\[([a-z0-9][a-z0-9-]*[a-z0-9])\]\]")


def extract_outgoing_links(content: str) -> list[str]:
    """Return all [[slug]] targets referenced in *content*.

    Args:
        content: Markdown page content.

    Returns:
        Deduplicated list of referenced page slugs.
    """
    return list(dict.fromkeys(_WIKI_LINK_RE.findall(content)))


def inject_backlink(content: str, source_slug: str) -> str:
    """Append a backlinks section to *content* if not already present.

    If a 'Backlinks' section already exists, insert *source_slug* into it.

    Args:
        content: Existing Markdown content of the target page.
        source_slug: Slug of the page that links to this page.

    Returns:
        Updated Markdown content with the backlink added.
    """
    raise NotImplementedError("Implemented in LW-13")
