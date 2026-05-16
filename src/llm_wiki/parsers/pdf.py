"""PDF text extractor with pypdf primary and pdfplumber fallback.

Strategy: try pypdf first (fast); if result is empty or garbled (<50 chars),
fall back to pdfplumber which handles tables and multi-column layouts better.
Implemented in LW-3.
"""

from pathlib import Path


class ParseError(Exception):
    """Raised when both PDF parsers fail to extract usable text."""


def parse_pdf(path: Path) -> str:
    """Extract plain text from a PDF file.

    Tries pypdf first for speed; falls back to pdfplumber on failure or
    when the extracted text appears garbled (fewer than 50 characters).

    Args:
        path: Absolute path to the PDF file.

    Returns:
        Extracted plain text (UTF-8, whitespace-normalised).

    Raises:
        ParseError: If both parsers fail to produce usable output.
    """
    text = _try_pypdf(path)
    if not text or len(text.strip()) < 50:
        text = _try_pdfplumber(path)
    if not text:
        raise ParseError(f"Both pypdf and pdfplumber failed for {path}")
    return text


def _try_pypdf(path: Path) -> str:
    """Attempt extraction with pypdf.

    Args:
        path: Path to the PDF file.

    Returns:
        Extracted text, or empty string on failure.
    """
    raise NotImplementedError("Implemented in LW-3")


def _try_pdfplumber(path: Path) -> str:
    """Attempt extraction with pdfplumber.

    Args:
        path: Path to the PDF file.

    Returns:
        Extracted text, or empty string on failure.
    """
    raise NotImplementedError("Implemented in LW-3")
