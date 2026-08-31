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
    "or photo (including handwritten notes) exactly as it appears — preserve "
    "reading order, line breaks, numbers, and render any tables as plain text. "
    "Do NOT translate, summarise, add commentary, or wrap the output in code "
    "fences. If the page has no legible text, reply with nothing."
)

# Vision API отклоняет слишком большие изображения; держим запас от лимита.
_MAX_IMAGE_BYTES = 20 * 1024 * 1024

# Фото конспектов/заметок, принимаемые как материалы (upload + пайплайн).
IMAGE_MIME_BY_EXT: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


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

    text = _transcribe_images(
        [("image/png", png) for png in pages], file_id=file_id
    )
    if not text:
        raise OCRError("OCR produced no text (blank or unreadable scan).")
    logger.info("ocr_done", file_id=file_id, pages=len(pages), chars=len(text))
    return text


def _transcribe_images(images: list[tuple[str, bytes]], file_id: str) -> str:
    """Vision-транскрипция изображений по порядку → сцепленный текст.

    Общее ядро для страниц скан-PDF (``ocr_pdf``) и фото-материалов
    (``ocr_image``). Каждый элемент — ``(mime, bytes)``.
    """
    import openai

    client = openai.OpenAI(
        api_key=settings.openai_api_key, timeout=settings.llm_timeout_s
    )
    logger.info(
        "ocr_start", file_id=file_id, model=settings.ocr_model, images=len(images)
    )

    texts: list[str] = []
    for idx, (mime, data) in enumerate(images, start=1):
        b64 = base64.b64encode(data).decode("ascii")
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
                                "image_url": {"url": f"data:{mime};base64,{b64}"},
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

    return "\n\n".join(texts).strip()


def ocr_image(path: Path, file_id: str = "ask") -> str:
    """Извлечь текст из фото/картинки (jpeg/png/webp) vision-моделью.

    Фича «фото конспектов»: юзер заливает снимок лекционных заметок (в т.ч.
    рукописных) как материал — vision-модель транскрибирует его в текст, дальше
    материал идёт по обычному пайплайну (вики-страница, поиск, артефакты).

    Raises:
        OCRError: нет API-ключа, неподдерживаемое расширение, файл слишком
            большой, сбой API или на снимке нет читаемого текста.
    """
    if not settings.openai_api_key:
        raise OCRError("OPENAI_API_KEY is required for OCR.")

    mime = IMAGE_MIME_BY_EXT.get(path.suffix.lower())
    if mime is None:
        raise OCRError(f"Unsupported image type: {path.suffix!r}")

    data = path.read_bytes()
    if len(data) > _MAX_IMAGE_BYTES:
        raise OCRError(
            "Изображение больше 20 МБ — сожмите фото и загрузите ещё раз."
        )
    if not data:
        raise OCRError("Image file is empty.")

    text = _transcribe_images([(mime, data)], file_id=file_id)
    if not text:
        raise OCRError(
            "На фото не найден читаемый текст — убедитесь, что снимок чёткий."
        )
    logger.info("ocr_image_done", file_id=file_id, chars=len(text))
    return text
