"""Markdown parser using markdown-it-py for AST-based extraction.

Extracts headings, wiki-links ([[slug]]), and plain text from Markdown files.
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

    Uses markdown-it-py's token stream for robust heading detection.
    All inline token content is concatenated into ``plain_text``;
    wiki-links (``[[slug]]``) are extracted from that text.

    Args:
        source: Raw Markdown string.

    Returns:
        ParsedMarkdown with headings, wiki-link slugs, and plain text.
    """
    if not source.strip():
        return ParsedMarkdown()

    from markdown_it import MarkdownIt  # local import — optional dep in tests

    md = MarkdownIt()
    tokens = md.parse(source)

    headings: list[Heading] = []
    inline_parts: list[str] = []

    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.type == "heading_open":
            level = int(token.tag[1:])  # "h1" → 1, "h2" → 2, …
            # The very next token in the stream is always `inline` for headings
            if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
                text = tokens[i + 1].content
                headings.append(Heading(level=level, text=text))
                inline_parts.append(text)
                i += 2  # consume both heading_open and its inline child
                continue
        elif token.type == "inline" and token.content:
            inline_parts.append(token.content)
        i += 1

    plain_text = "\n".join(inline_parts)
    links = extract_wiki_links(plain_text)

    return ParsedMarkdown(headings=headings, links=links, plain_text=plain_text)


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
    """Return all [[slug]] references found in *text*, deduplicated.

    Preserves first-seen order while removing duplicates.

    Args:
        text: Raw Markdown string.

    Returns:
        Deduplicated list of slug strings (the text inside the double brackets).
    """
    return list(dict.fromkeys(_WIKI_LINK_RE.findall(text)))
