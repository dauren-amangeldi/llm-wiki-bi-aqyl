"""Unit tests for PDF and Markdown parsers."""

from pathlib import Path
from unittest.mock import patch

import pytest

from llm_wiki.parsers.markdown import ParsedMarkdown, extract_wiki_links, parse_markdown
from llm_wiki.parsers.pdf import ParseError, parse_pdf


# ===========================================================================
# Markdown parser
# ===========================================================================


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

    def test_parse_markdown_headings(self) -> None:
        """parse_markdown should extract all headings with correct levels."""
        source = "# Title\n## Section\n### Subsection\n"
        result = parse_markdown(source)
        assert len(result.headings) == 3
        assert result.headings[0].level == 1
        assert result.headings[0].text == "Title"
        assert result.headings[1].level == 2
        assert result.headings[2].level == 3

    def test_parse_markdown_plain_text(self) -> None:
        """parse_markdown plain_text should contain the body text."""
        source = "# Title\n\nSome body text here.\n"
        result = parse_markdown(source)
        assert "Some body text here." in result.plain_text

    def test_parse_markdown_extracts_wiki_links(self) -> None:
        """parse_markdown should populate the links field from [[slug]] refs."""
        source = "See [[bert]] and [[gpt]] for details.\n"
        result = parse_markdown(source)
        assert "bert" in result.links
        assert "gpt" in result.links

    def test_parse_markdown_empty_input(self) -> None:
        """parse_markdown with empty or whitespace input returns empty ParsedMarkdown."""
        empty = parse_markdown("")
        assert empty.headings == []
        assert empty.links == []
        assert empty.plain_text == ""

        whitespace = parse_markdown("   \n\n  ")
        assert whitespace.headings == []

    def test_parse_markdown_nested_headings(self) -> None:
        """parse_markdown handles deeply nested heading levels."""
        source = "# H1\n## H2\n### H3\n#### H4\n"
        result = parse_markdown(source)
        assert [h.level for h in result.headings] == [1, 2, 3, 4]

    def test_parse_markdown_links_deduplicated(self) -> None:
        """parse_markdown links field should not contain duplicates."""
        source = "[[foo]] is mentioned twice [[foo]]."
        result = parse_markdown(source)
        assert result.links.count("foo") == 1


# ===========================================================================
# PDF parser
# ===========================================================================


class TestPdfParser:
    """Tests for the PDF parser."""

    def test_parse_pdf_returns_text(self, sample_pdf_path: Path) -> None:
        """parse_pdf should return non-empty text from a valid PDF."""
        result = parse_pdf(sample_pdf_path)
        assert isinstance(result, str)
        assert len(result.strip()) > 50

    def test_parse_pdf_contains_known_content(self, sample_pdf_path: Path) -> None:
        """parse_pdf result should contain text we know is in the fixture PDF."""
        result = parse_pdf(sample_pdf_path)
        # The fixture PDF contains 'Transformer' — verify at least one known word
        assert "Transformer" in result or "transformer" in result.lower()

    def test_parse_pdf_fallback_to_pdfplumber(self, sample_pdf_path: Path) -> None:
        """When pypdf returns empty text, pdfplumber should take over."""
        with patch("llm_wiki.parsers.pdf._try_pypdf", return_value=""):
            result = parse_pdf(sample_pdf_path)
        assert len(result.strip()) > 50

    def test_parse_pdf_raises_on_both_failures(self, tmp_path: Path) -> None:
        """ParseError is raised when both pypdf and pdfplumber fail."""
        dummy = tmp_path / "bad.pdf"
        dummy.write_bytes(b"not a real pdf")
        with pytest.raises(ParseError):
            parse_pdf(dummy)

    def test_try_pypdf_returns_empty_on_exception(self, tmp_path: Path) -> None:
        """_try_pypdf returns empty string on any exception (not raises)."""
        from llm_wiki.parsers.pdf import _try_pypdf

        bad = tmp_path / "bad.pdf"
        bad.write_bytes(b"garbage")
        result = _try_pypdf(bad)
        assert result == ""

    def test_try_pdfplumber_returns_empty_on_exception(self, tmp_path: Path) -> None:
        """_try_pdfplumber returns empty string on any exception (not raises)."""
        from llm_wiki.parsers.pdf import _try_pdfplumber

        bad = tmp_path / "bad.pdf"
        bad.write_bytes(b"garbage")
        result = _try_pdfplumber(bad)
        assert result == ""
