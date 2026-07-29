"""OCR for scanned/image PDFs via the OpenAI vision API.

When a PDF has no text layer (a scan or a photo of a page), pypdf/pdfplumber
extract nothing. This renders each page to an image (pypdfium2 — no system
deps) and reads the text with a vision model. Same "API-based extraction"
pattern as ``audio.transcribe_audio``, using a synchronous OpenAI client so it
is callable from the synchronous ingestion step.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import structlog

from llm_wiki.config import settings

logger = structlog.get_logger(__name__)

_OCR_PROMPT = (
    "You are an OCR engine. Transcribe ALL text visible on this document page "
    "exactly as it appears — preserve reading order, line breaks, numbers, and "
    "render any tables as plain text. Do NOT translate, summarise, add "
    "commentary, or wrap the output in code fences. If the page has no legible "
    "text, reply with nothing."
)


class OCRError(Exception):
    """Raised when a scanned PDF cannot be OCR'd."""


def _render_pages_png(path: Path, max_pages: int, scale: float) -> list[bytes]:
    """Render up to *max_pages* PDF pages to PNG bytes with pypdfium2."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    try:
        count = min(len(pdf), max_pages)
        pngs: list[bytes] = []
        for i in range(count):
            page = pdf[i]
            bitmap = page.render(scale=scale)
            image = bitmap.to_pil()
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            pngs.append(buf.getvalue())
            page.close()
        return pngs
    finally:
        pdf.close()


def ocr_pdf(path: Path, file_id: str = "ask") -> str:
    """Extract text from a scanned/image PDF using the OpenAI vision API.

    Renders each page (capped at ``settings.ocr_max_pages``) and asks a
    vision model to transcribe it, concatenating pages in order.

    Args:
        path: Absolute path to the PDF file.
        file_id: Correlation ID for structured logging.

    Returns:
        The transcribed text across all rendered pages.

    Raises:
        OCRError: If the API key is missing, rendering fails, the API call
            fails, or no text was produced across all pages.
    """
    if not settings.openai_api_key:
        raise OCRError("OPENAI_API_KEY is required for OCR.")

    try:
        pages = _render_pages_png(
            path,
            max_pages=settings.ocr_max_pages,
            scale=settings.ocr_render_scale,
        )
    except Exception as exc:  # noqa: BLE001 - surface any render/decode error
        raise OCRError(f"Failed to render PDF for OCR: {exc}") from exc
    if not pages:
        raise OCRError("PDF has no pages to OCR.")

    import openai

    client = openai.OpenAI(
        api_key=settings.openai_api_key, timeout=settings.llm_timeout_s
    )
    logger.info("ocr_start", file_id=file_id, model=settings.ocr_model, pages=len(pages))

    texts: list[str] = []
    for idx, png in enumerate(pages, start=1):
        b64 = base64.b64encode(png).decode("ascii")
        try:
            resp = client.chat.completions.create(
                model=settings.ocr_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": _OCR_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"},
                            },
                        ],
                    }
                ],
            )
        except Exception as exc:  # noqa: BLE001 - surface any OpenAI/transport error
            logger.error("ocr_page_failed", file_id=file_id, page=idx, error=str(exc))
            raise OCRError(f"OCR failed on page {idx}: {exc}") from exc
        page_text = (resp.choices[0].message.content or "").strip()
        if page_text:
            texts.append(page_text)

    text = "\n\n".join(texts).strip()
    if not text:
        raise OCRError("OCR produced no text (blank or unreadable scan).")
    logger.info("ocr_done", file_id=file_id, pages=len(pages), chars=len(text))
    return text
