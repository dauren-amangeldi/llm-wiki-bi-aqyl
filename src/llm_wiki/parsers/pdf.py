"""PDF text extractor with pypdf primary and pdfplumber fallback.

Strategy: try pypdf first (fast); if result is empty or garbled (<50 chars),
fall back to pdfplumber which handles tables and multi-column layouts better.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


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
        logger.warning("pypdf produced insufficient text, falling back to pdfplumber", path=str(path))
        text = _try_pdfplumber(path)
    if not text:
        raise ParseError(f"Both pypdf and pdfplumber failed for {path}")
    return text


def _try_pypdf(path: Path) -> str:
    """Attempt extraction with pypdf.

    Args:
        path: Path to the PDF file.

    Returns:
        Extracted plain text stripped of leading/trailing whitespace,
        or an empty string on any failure.
    """
    try:
        import pypdf  # local import — avoids hard dep at module level

        reader = pypdf.PdfReader(str(path))
        pages_text: list[str] = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                pages_text.append(page_text)
        return "\n".join(pages_text).strip()
    except Exception as exc:  # noqa: BLE001
        logger.debug("pypdf failed", path=str(path), error=str(exc))
        return ""


def _try_pdfplumber(path: Path) -> str:
    """Attempt extraction with pdfplumber.

    Args:
        path: Path to the PDF file.

    Returns:
        Extracted plain text stripped of leading/trailing whitespace,
        or an empty string on any failure.
    """
    try:
        import pdfplumber  # local import — avoids hard dep at module level

        with pdfplumber.open(str(path)) as pdf:
            pages_text: list[str] = []
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    pages_text.append(page_text)
        return "\n".join(pages_text).strip()
    except Exception as exc:  # noqa: BLE001
        logger.debug("pdfplumber failed", path=str(path), error=str(exc))
        return ""
