"""Markdown parser using markdown-it-py for AST-based extraction.

Extracts headings, wiki-links ([[slug]]), and plain text from Markdown files.
Implemented in LW-3.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Heading:
    """A heading entry extracted from a Markdown document."""

    level: int
    text: str


@dataclass
class ParsedMarkdown:
    """Structured result of parsing a Markdown document."""

    headings: list[Heading] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    plain_text: str = ""


def parse_markdown(source: str) -> ParsedMarkdown:
    """Parse Markdown source into headings, links, and plain text.

    Args:
        source: Raw Markdown string.

    Returns:
        ParsedMarkdown with headings, wiki-link slugs, and plain text.
    """
    raise NotImplementedError("Implemented in LW-3")


def parse_markdown_file(path: Path) -> ParsedMarkdown:
    """Read a Markdown file and parse it.

    Args:
        path: Path to the .md file.

    Returns:
        ParsedMarkdown extracted from the file contents.
    """
    source = path.read_text(encoding="utf-8")
    return parse_markdown(source)


_WIKI_LINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


def extract_wiki_links(text: str) -> list[str]:
    """Return all [[slug]] references found in *text*.

    Args:
        text: Raw Markdown string.

    Returns:
        List of slug strings (the text inside the double brackets).
    """
    return _WIKI_LINK_RE.findall(text)
