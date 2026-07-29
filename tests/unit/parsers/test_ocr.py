"""Unit tests for the scanned-PDF OCR fallback (renders real pages, mocks OpenAI)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_wiki.config import settings
from llm_wiki.parsers.ocr import OCRError, ocr_pdf


def _image_pdf(tmp_path: Path, pages: int = 1) -> Path:
    """Build an image-only PDF (no text layer) so ocr_pdf's renderer has real input."""
    from PIL import Image, ImageDraw

    imgs = []
    for i in range(pages):
        img = Image.new("RGB", (600, 300), "white")
        ImageDraw.Draw(img).text((20, 140), f"page {i}", fill="black")
        imgs.append(img)
    out = tmp_path / "scan.pdf"
    imgs[0].save(str(out), "PDF", save_all=True, append_images=imgs[1:])
    return out


def _mock_openai(page_texts: list[str]) -> MagicMock:
    """Fake openai.OpenAI returning page_texts from chat.completions.create in order."""
    seq = iter(page_texts)

    def create(**_kw: object) -> MagicMock:
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content=next(seq)))]
        return resp

    client = MagicMock()
    client.chat.completions.create.side_effect = create
    return client


def test_ocr_pdf_concatenates_pages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    pdf = _image_pdf(tmp_path, pages=2)
    fake = _mock_openai(["Страница один", "Страница два"])
    with patch("openai.OpenAI", return_value=fake):
        out = ocr_pdf(pdf, file_id="t")
    assert out == "Страница один\n\nСтраница два"


def test_ocr_skips_blank_pages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    pdf = _image_pdf(tmp_path, pages=2)
    fake = _mock_openai(["", "Только вторая"])  # first page unreadable
    with patch("openai.OpenAI", return_value=fake):
        out = ocr_pdf(pdf, file_id="t")
    assert out == "Только вторая"


def test_ocr_respects_max_pages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    monkeypatch.setattr(settings, "ocr_max_pages", 1)
    pdf = _image_pdf(tmp_path, pages=3)
    fake = _mock_openai(["only one call"])  # would StopIteration if it OCR'd >1 page
    with patch("openai.OpenAI", return_value=fake):
        out = ocr_pdf(pdf, file_id="t")
    assert out == "only one call"


def test_ocr_requires_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openai_api_key", "")
    pdf = _image_pdf(tmp_path)
    with pytest.raises(OCRError, match="API_KEY"):
        ocr_pdf(pdf)


def test_ocr_raises_when_no_text_produced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    pdf = _image_pdf(tmp_path)
    fake = _mock_openai([""])  # vision returns nothing on every page
    with patch("openai.OpenAI", return_value=fake):
        with pytest.raises(OCRError, match="no text"):
            ocr_pdf(pdf)


def test_ocr_wraps_api_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    pdf = _image_pdf(tmp_path)
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("boom")
    with patch("openai.OpenAI", return_value=client):
        with pytest.raises(OCRError, match="page 1"):
            ocr_pdf(pdf)
