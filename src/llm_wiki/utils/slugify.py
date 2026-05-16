"""Convert human-readable titles into URL-safe wiki page slugs."""

import re

from slugify import slugify as _slugify


def to_slug(title: str) -> str:
    """Convert a page title to a kebab-case wiki slug.

    Uses python-slugify to handle Unicode, punctuation, and spacing.

    Args:
        title: Human-readable page title, e.g. 'Transformer Architecture (2017)'.

    Returns:
        Lowercase kebab-case slug, e.g. 'transformer-architecture-2017'.
    """
    return _slugify(title, separator="-", lowercase=True, max_length=80)


def is_valid_slug(slug: str) -> bool:
    """Return True if *slug* is a valid wiki page identifier.

    A valid slug contains only lowercase ASCII letters, digits, and hyphens.

    Args:
        slug: Candidate slug string.
    """
    return bool(re.fullmatch(r"[a-z0-9][a-z0-9-]*[a-z0-9]", slug))
