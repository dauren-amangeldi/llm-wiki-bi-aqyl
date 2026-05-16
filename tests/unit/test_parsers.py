"""Unit tests for PDF and Markdown parsers.

Full implementation added in LW-3. These tests act as acceptance criteria.
"""

import pytest

from llm_wiki.parsers.markdown import ParsedMarkdown, extract_wiki_links, parse_markdown


class TestMarkdownParser:
    """Tests for the Markdown parser."""

    def test_extract_wiki_links_basic(self) -> None:
        """extract_wiki_links should find [[slug]] references."""
        text = "See [[transformers]] and [[attention-mechanism]] for more."
        links = extract_wiki_links(text)
        assert "transformers" in links
        assert "attention-mechanism" in links

    def test_extract_wiki_links_no_links(self) -> None:
        """extract_wiki_links returns empty list when no wiki links exist."""
        assert extract_wiki_links("No links here.") == []

    def test_extract_wiki_links_deduplicates(self) -> None:
        """extract_wiki_links should not return duplicate slugs."""
        text = "[[foo]] and [[foo]] again"
        links = extract_wiki_links(text)
        assert links.count("foo") == 1

    @pytest.mark.xfail(reason="Implemented in LW-3")
    def test_parse_markdown_headings(self) -> None:
        """parse_markdown should extract all headings with correct levels."""
        source = "# Title\n## Section\n### Subsection\n"
        result = parse_markdown(source)
        assert len(result.headings) == 3
        assert result.headings[0].level == 1
        assert result.headings[0].text == "Title"

    @pytest.mark.xfail(reason="Implemented in LW-3")
    def test_parse_markdown_plain_text(self) -> None:
        """parse_markdown plain_text should contain the body text."""
        source = "# Title\n\nSome body text here.\n"
        result = parse_markdown(source)
        assert "Some body text here." in result.plain_text


class TestPdfParser:
    """Tests for the PDF parser — require fixture files (added in LW-3)."""

    @pytest.mark.xfail(reason="Fixture file added in LW-3")
    def test_parse_pdf_returns_text(self, sample_pdf_path: object) -> None:
        """parse_pdf should return non-empty text from a valid PDF."""
        from llm_wiki.parsers.pdf import parse_pdf

        result = parse_pdf(sample_pdf_path)  # type: ignore[arg-type]
        assert isinstance(result, str)
        assert len(result.strip()) > 50
